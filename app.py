from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import json
from dotenv import load_dotenv
import base64
import uuid
import time
import random
import io
import asyncio
import aiohttp
from PIL import Image
from moviepy import ImageClip, TextClip, CompositeVideoClip, ColorClip, concatenate_videoclips

load_dotenv()

app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'temp'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Stable Horde API configuration
STABLE_HORDE_API = "https://stablehorde.net/api/v2"
# Optional: Get an API key from https://stablehorde.net/register to get priority in the queue
STABLE_HORDE_API_KEY = os.getenv('STABLE_HORDE_API_KEY')

# Default settings for image generation
DEFAULT_IMAGE_SETTINGS = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg_scale": 7.5,
    "karras": True,
    "post_processing": ["GFPGAN"],
    "clip_skip": 1,
    "models": ["Deliberate"],  # Using Deliberate model for better quality
    "negative_prompt": "blurry, low quality, low resolution, watermark, signature, text, cropped, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, out of frame, cartoon, 3d, cgi, render, painting, drawing, sketch"
}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/synthesize_dream', methods=['POST'])
def synthesize_dream():
    dream_text = request.json.get('dream')
    chat_history = []
    prompt = f"""
            Decompose the following dream narration into coherent scenes with clear descriptions.
            For each scene, extract the following:
            - "scene_id": A unique integer ID for the scene (1, 2, 3).
            - "description": A concise summary of the scene.
            - "mood": The emotional tone or atmosphere of the scene (e.g., "surreal", "whimsical threat", "calm", "anxious").
            - "visual_tags": A list of keywords or phrases that describe visual elements suitable for image generation (e.g., ["ceiling", "jelly", "walking", "dreamlike colors"]).

            Return the output as a JSON array of objects.

            Dream: "{dream_text}"
    """
    
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    decompose_payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "scene_id": {"type": "NUMBER"},
                        "description": {"type": "STRING"},
                        "mood": {"type": "STRING"},
                        "visual_tags": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        }
                    },
                    "required": ["scene_id", "description", "mood", "visual_tags"]
                }
            }
        }
    }
    
    decompose_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    decompose_response = requests.post(decompose_api_url, headers={'Content-Type': 'application/json'}, json=decompose_payload)
    decompose_response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    decompose_result = decompose_response.json()

    # Extract scene data from decomposition response
    if decompose_result.get('candidates') and decompose_result['candidates'][0].get('content') and decompose_result['candidates'][0]['content'].get('parts'):
        json_string = decompose_result['candidates'][0]['content']['parts'][0]['text']
        scenes_data = json.loads(json_string)
    else:
        raise ValueError("Failed to get valid scene decomposition response from LLM.")
    
    processed_scenes = []
    for scene_data in scenes_data:  # Fixed variable name
        image_url = generate_image_for_scene(scene_data)
        caption = generate_caption_for_scene(scene_data)
        processed_scenes.append({
            "scene_id": scene_data['scene_id'],
            "description": scene_data['description'],
            "mood": scene_data['mood'],
            "visual_tags": scene_data['visual_tags'],
            "image_url": image_url,
            "caption": caption
        })
    
    # Only create video if we have valid images
    valid_scenes = [scene for scene in processed_scenes if scene['image_url'] is not None]
    if valid_scenes:
        video_url = create_video_montage(valid_scenes, app.config['UPLOAD_FOLDER'])
    else:
        video_url = None
        
    return jsonify({
        "scenes": processed_scenes,
        "video_url": video_url
    })

def make_api_request_with_retry(api_url, payload, max_retries=3, is_image_generation=False):
    """Helper function to make API requests with retry logic"""
    retry_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Add jitter and exponential backoff
            if attempt > 0:
                delay = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} of {max_retries}...")
                print(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            
            print(f"Making {'image generation' if is_image_generation else 'API'} request to: {api_url}")
            if not is_image_generation:
                print(f"Request payload: {json.dumps(payload, indent=2)[:500]}...")  # Log first 500 chars of payload
                
            start_time = time.time()
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=60  # Increased timeout for image generation
            )
            elapsed = time.time() - start_time
            
            print(f"Response status: {response.status_code} (took {elapsed:.2f}s)")
            
            # Handle rate limiting and other HTTP errors
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', retry_delay))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            # Log error response details
            if response.status_code >= 400:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(error_msg)
                if response.status_code >= 500:
                    # For server errors, retry
                    continue
                else:
                    # For client errors, fail fast
                    raise RuntimeError(error_msg)
            
            response_data = response.json()
            if is_image_generation:
                print("Image generation response received")
            return response_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(error_msg)
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request failed after {max_retries} attempts: {error_msg}")
            print(f"Attempt {attempt + 1} failed, retrying...")
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            print(error_msg)
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to parse API response after {max_retries} attempts: {error_msg}")
    
    raise RuntimeError(f"Failed to complete API request after {max_retries} attempts")

async def generate_image_with_stable_horde(prompt, settings):
    """Generate an image using Stable Horde's API"""
    headers = {
        "Content-Type": "application/json",
        "apikey": STABLE_HORDE_API_KEY,
        "Client-Agent": "DreamVisualiser/1.0"
    }
    
    # Prepare the payload
    payload = {
        "prompt": f"{prompt}, high quality, highly detailed, dreamlike, surreal, artistic, cinematic lighting, 8k",
        **settings,
        "n": 1,
        "seed": random.randint(0, 2**32 - 1)
    }
    
    # Submit the generation request
    async with aiohttp.ClientSession() as session:
        # Start the generation
        async with session.post(
            f"{STABLE_HORDE_API}/generate/async",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 202:
                error = await response.text()
                raise ValueError(f"Failed to start generation: {error}")
            
            result = await response.json()
            if not result.get('id'):
                raise ValueError("No generation ID received")
            
            generation_id = result['id']
            print(f"Generation started with ID: {generation_id}")
            
            # Poll for results
            max_attempts = 150  # 5 minutes max (5s * 60)
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Check every 5 seconds
                print(f"Checking generation status... ({attempt + 1}/{max_attempts})")
                
                try:
                    async with session.get(
                        f"{STABLE_HORDE_API}/generate/check/{generation_id}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as check_response:
                        if check_response.status != 200:
                            print(f"Status check failed: {check_response.status}")
                            continue
                        
                        status = await check_response.json()
                        print(f"Status: {status}")
                        
                        if status.get('done', False):
                            print("Generation completed!")
                            break
                        elif status.get('faulted', False):
                            raise ValueError(f"Generation failed: {status.get('message', 'Unknown error')}")
                            
                except Exception as e:
                    print(f"Error checking status: {e}")
                    continue
            else:
                raise TimeoutError("Image generation timed out")
            
            # Get the generated image
            async with session.get(
                f"{STABLE_HORDE_API}/generate/status/{generation_id}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as status_response:
                if status_response.status != 200:
                    error = await status_response.text()
                    raise ValueError(f"Failed to get generation status: {error}")
                
                result = await status_response.json()
                print(f"Final result: {result}")
                
                if not result.get('generations') or not result['generations']:
                    raise ValueError("No generations in response")
                
                # Get the first image
                generation = result['generations'][0]
                image_b64 = generation.get('img')
                if not image_b64:
                    raise ValueError("No image data in response")
                
                # Handle base64 data with or without data URI prefix
                if image_b64.startswith('data:'):
                    image_b64 = image_b64.split(',', 1)[1]
                
                return base64.b64decode(image_b64)

def generate_image_for_scene(scene_data):
    """Generate an image for a scene using Stable Horde"""
    try:
        print(f"\nStarting image generation for scene: {scene_data.get('scene_id', 'unknown')}")
        description = scene_data.get('description', '')
        mood = scene_data.get('mood', 'mysterious')
        visual_tags = " ".join(scene_data.get('visual_tags', []))
        
        # Step 1: Generate image prompt
        prompt_gen_prompt = f"""
        Create a detailed, creative text-to-image prompt based on this dream scene:
        Description: "{description}"
        Mood: "{mood}"
        Visual Elements: "{visual_tags}"
        
        Focus on creating a surreal, dreamlike atmosphere. Be specific about colors, lighting, and composition.
        Keep the prompt under 300 characters.
        """
        
        print("\nGenerating image prompt...")
        prompt_result = make_api_request_with_retry(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
            {
                "contents": [{"parts": [{"text": prompt_gen_prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.8,
                    "topK": 32,
                    "maxOutputTokens": 300
                }
            }
        )
        
        # Extract the generated prompt
        image_prompt = ""
        if 'candidates' in prompt_result and prompt_result['candidates']:
            candidate = prompt_result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                for part in candidate['content']['parts']:
                    if 'text' in part:
                        image_prompt = part['text'].strip()
        
        if not image_prompt:
            image_prompt = f"A dreamlike scene showing: {description}. Mood: {mood}. Visual elements: {visual_tags}"
        
        # Ensure the prompt isn't too long
        image_prompt = image_prompt[:300]
        print(f"Generated prompt: {image_prompt}")
        
        # Step 2: Generate the image using Stable Horde
        print("\nGenerating image with Stable Horde...")
        
        # Run the async function in the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            image_bytes = loop.run_until_complete(
                generate_image_with_stable_horde(image_prompt, DEFAULT_IMAGE_SETTINGS)
            )
        except Exception as e:
            print(f"Error in async image generation: {e}")
            # Try to close the loop if we created it
            if loop.is_running():
                loop.stop()
            raise
        
        # Save the image
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        print(f"Image saved to: {image_path}")
        return f"/temp/{image_filename}"
            
    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        # Return None if generation fails
        return None

def generate_caption_for_scene(scene_data):
    """
    Generates a short, poetic caption for a dream scene using Gemini-2.5-Flash.
    """
    description = scene_data['description']
    mood = scene_data['mood']

    # --- LLM Call for Caption Generation (gemini-2.5-flash) ---
    prompt = f"""
    Generate a short, poetic, and surreal one-liner caption for a dream scene.
    The scene is described as: "{description}"
    The mood is: "{mood}"

    Ensure the caption captures the essence of a dream, being slightly cryptic or evocative.
    """
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.8,
            "topK": 32,
            "maxOutputTokens": 100
        }
    }
    
    try:
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            caption = result['candidates'][0]['content']['parts'][0]['text'].strip()
            return caption
        else:
            # Fallback caption if LLM response is not as expected
            return "A dream's echo, softly spoken."

    except requests.exceptions.RequestException as e:
        print(f"Error generating caption: {e}. Using fallback.")
        return "A dream's echo, softly spoken."
    except Exception as e:
        print(f"Error processing caption response: {e}. Using fallback.")
        return "A dream's echo, softly spoken."

def create_video_montage(scenes_with_media, output_folder):
    """
    Creates a video montage from a list of scenes, each with an image and a caption.
    Uses MoviePy to compose the video and saves it temporarily.
    """
    clips = []
    scene_duration = 5  # seconds per scene for each image
    text_color = "white"
    font_path = "Arial"  # Common font name, MoviePy might resolve this or use default

    for scene_media in scenes_with_media:
        # Skip scenes without valid image URLs
        if not scene_media.get('image_url'):
            print(f"Skipping scene {scene_media.get('scene_id', 'unknown')} - no image URL")
            continue
            
        # Convert the URL path to a local file system path
        image_filename = os.path.basename(scene_media['image_url'])
        image_path = os.path.join(output_folder, image_filename)
        caption = scene_media.get('caption', 'A dream unfolds...')

        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        try:
            # Create an ImageClip from the generated image
            img_clip = ImageClip(image_path, duration=scene_duration)

            # Create a TextClip for the caption
            # Position the text at the bottom center of the clip
            txt_clip = TextClip(caption, fontsize=40, color=text_color, font=font_path,
                                stroke_color='black', stroke_width=1.5)
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(scene_duration)

            # If the text is too wide, resize it to fit within the image boundaries
            if txt_clip.w > img_clip.w * 0.9:
                txt_clip = txt_clip.resize(width=img_clip.w * 0.9)

            # Composite the image and text clips
            final_clip = CompositeVideoClip([img_clip, txt_clip])
            clips.append(final_clip)
            print(f"Successfully created clip for scene {scene_media.get('scene_id')}")
            
        except Exception as e:
            print(f"Error creating clip for scene {scene_media.get('scene_id')}: {e}")
            # Create a black screen as fallback
            blank_clip = ColorClip(size=(1024, 1024), color=(0, 0, 0), duration=scene_duration)
            # Replace your current TextClip line with:
            try:
                text_clip = TextClip("Scene could not be rendered", 
                                    font_size=40, 
                                    color="white",
                                    duration=scene_duration,
                                    font="Arial")  # Specify a common font
            except:
                # Fallback without font specification
                text_clip = TextClip("Scene could not be rendered", 
                                    font_size=40, 
                                    color="white",
                                    duration=scene_duration)
            text_clip = text_clip.set_position('center').set_duration(scene_duration)
            fallback_clip = CompositeVideoClip([blank_clip, text_clip])
            clips.append(fallback_clip)

    if not clips:
        raise ValueError("No valid video clips could be created for the montage.")

    # Concatenate all scene clips into a final video
    final_video_clip = concatenate_videoclips(clips)
    output_filename = f"dream_montage_{uuid.uuid4()}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # Write the final video file
    final_video_clip.write_videofile(
        output_path, 
        fps=24, 
        codec="libx264", 
        audio_codec="aac", 
        bitrate="1500k", 
        preset="fast",
        verbose=False,
        logger=None
    )
    
    # Clean up clips to free memory
    for clip in clips:
        clip.close()
    final_video_clip.close()
    
    return f"/temp/{output_filename}"

@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """
    Serves temporary files (images and videos) from the 'temp' directory.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)