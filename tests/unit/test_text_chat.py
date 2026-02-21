import pytest
import os
import logging
from unittest.mock import AsyncMock
import base64

# Adjust path to import project modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from main_logic.omni_offline_client import OmniOfflineClient

logger = logging.getLogger(__name__)

# Dummy 1x1 pixel PNG image in base64
DUMMY_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKwjwAAAAABJRU5ErkJggg=="

@pytest.fixture
def offline_client():
    """Returns an OmniOfflineClient instance configured with Qwen (default)."""
    from utils.api_config_loader import get_assist_api_profiles
    assist_profiles = get_assist_api_profiles()
    
    # Use Qwen as the standard test provider if available, else OpenAI
    provider = "qwen" if "qwen" in assist_profiles else "openai"
    if provider not in assist_profiles:
        pytest.skip("No Qwen or OpenAI profile found for testing.")
        
    profile = assist_profiles[provider]
    
    api_key = profile.get('OPENROUTER_API_KEY')
    if not api_key:
        # Fallback for Qwen/OpenAI
        env_key = "ASSIST_API_KEY_QWEN" if provider == "qwen" else "ASSIST_API_KEY_OPENAI"
        api_key = os.environ.get(env_key)
        
    if not api_key:
        pytest.skip(f"API key for {provider} not found.")

    client = OmniOfflineClient(
        base_url=profile['OPENROUTER_URL'],
        api_key=api_key,
        model=profile['CORRECTION_MODEL'], # Use correction model as it is usually a chat model
        vision_model=profile.get('VISION_MODEL', ''),
        vision_base_url=profile.get('VISION_BASE_URL', ''),
        vision_api_key=profile.get('VISION_API_KEY', ''),
        on_text_delta=AsyncMock(),
        on_response_done=AsyncMock()
    )
    return client

@pytest.mark.unit
async def test_simple_text_chat(offline_client, llm_judger):
    """Test sending a simple text message and checking the response quality."""
    prompt = "Tell me a very short joke with less than 20 words."
    
    # OmniOfflineClient uses callbacks. We need to capture the output from on_text_delta.
    response_accumulator = []
    
    async def on_text_delta(text, is_first):
        response_accumulator.append(text)
        
    # Replace the MagicMock with our capturing function
    offline_client.on_text_delta = on_text_delta
    
    logger.info(f"Sending prompt: {prompt}")
    
    try:
        # In OmniOfflineClient, we usually append the user message to history then call create_response
        # But create_response takes 'instructions' which might be treated as system prompt or user prompt depending on impl.
        # Looking at code: create_response(instructions) -> appends as user message if not system prefix.
        
        # Actually stream_text(text) is the main method for user input text.
        await offline_client.stream_text(prompt)
        
        full_response = "".join(response_accumulator)
        logger.info(f"Received response: {full_response}")
        
        assert len(full_response) > 0, "Response should not be empty"
        
        # Verify with LLM Judger
        passed = llm_judger.judge(
            input_text=prompt,
            output_text=full_response,
            criteria="Is this a joke? Is it short (under 50 words)?"
        )
        assert passed, f"LLM Judger rejected the response: {full_response}"
        
    except Exception as e:
        pytest.fail(f"Text chat failed: {e}")



@pytest.mark.unit
async def test_vision_chat(offline_client, llm_judger):
    """Test sending an image and asking for a description."""
    if not offline_client.vision_model:
        # Check if model itself supports vision (like gpt-4o) if vision_model is not explicitly set separate
         pass

    # Read the actual test image
    image_path = os.path.join(os.path.dirname(__file__), '../test_inputs/screenshot.png')
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")
        
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = "What is in this image? Describe it briefly."
    keywords = ["steam", "n.e.k.o.", "girl", "character", "猫娘"]
    
    response_accumulator = []
    async def on_text_delta(text, is_first):
        response_accumulator.append(text)
    
    offline_client.on_text_delta = on_text_delta
    
    logger.info(f"Sending vision prompt with image: {image_path}")
    
    try:
        # OOC workflow: stream_image() (adds to pending) then stream_text() (sends pending + text)
        await offline_client.stream_image(image_b64)
        await offline_client.stream_text(prompt)
        
        full_response = "".join(response_accumulator)
        logger.info(f"Received vision response: {full_response}")
        
        assert len(full_response) > 0
        
        # Validation 1: fast keyword check
        request_verification = any(k.lower() in full_response.lower() for k in keywords)
        
        if request_verification:
             logger.info("✅ Keyword validation passed locally.")
        else:
             logger.warning(f"⚠️ Keywords {keywords} not found in response. Fallback to LLM identification.")

        # Validation 2: LLM Judger for semantic correctness
        # "这个图片的关键词是steam，N.E.K.O.，girl——答到一个就可以。"
        criteria = (
            "The user provided an image of a software interface or game character. "
            "Does the response mention 'Steam', 'N.E.K.O.', a girl/character, or imply seeing a game library/store page? "
            "Answer YES if ANY of these are mentioned or described."
        )
        
        passed = llm_judger.judge(
            input_text=f"{prompt} [Image Provided]",
            output_text=full_response,
            criteria=criteria
        )
        assert passed, f"LLM Judger rejected vision response: {full_response}"
        
    except Exception as e:
        pytest.fail(f"Vision chat failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
