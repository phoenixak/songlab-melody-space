import io
import json
import os
import time
import uuid
from datetime import datetime
from io import BytesIO
from tempfile import NamedTemporaryFile
from xmlrpc.client import Binary
import jwt
import threading

import numpy as np
import requests
import scipy
import soundfile as sf
import streamlit as st
import streamlit_vertical_slider as svs
from pydub import AudioSegment
from scipy.signal import butter, sosfilt
from streamlit import session_state as st_state
from woocommerce import API
from wordpress_xmlrpc import Client
from wordpress_xmlrpc.compat import xmlrpc_client
from wordpress_xmlrpc.methods import media

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Try to get API_URL from environment variables, if not found set to a default value
try:
    API_URL = os.environ["API_URL"]
except KeyError:
    st.error("API_URL environment variable is not set.")
    st.stop()
    
    # Try to get the Bearer token from environment variables, if not found set to a default value
try:
    BEARER_TOKEN = os.environ["BEARER_TOKEN"]
except KeyError:
    st.error("BEARER_TOKEN environment variable is not set.")
    st.stop()

print("API_URL:", os.environ["API_URL"])
print("BEARER_TOKEN:", os.environ["BEARER_TOKEN"])



page_bg_img = '''
<style>
.stApp {
background-image: url("https://songlabai.com/wp-content/uploads/2024/03/4.png");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)



def save_to_wordpress(audio_array, sample_rate):
    # Convert audio_array to float32 if not already
    audio_array = audio_array.astype(np.float32)

    # Save the audio to a BytesIO buffer
    wav_bytes = BytesIO()
    sf.write(wav_bytes, audio_array, samplerate=sample_rate, format='WAV')
    wav_bytes.seek(0)

    # Define your WordPress site URL and authentication credentials
    wordpress_url = "https://songlabai.com/xmlrpc.php"
    woocommerce_url = "https://songlabai.com"
    consumer_key = "ck_93d516ba12289a6fd0eced56bbc0b05ecbf98735"
    consumer_secret = "cs_9d5eb716d631db408a4c47796b5d18b0313d8559"
    username = "admin_h2ibbgql"
    password = "Pressabc1!"


    
    # Authenticate with WordPress XML-RPC API
    title = f"generated_audio_{datetime.now().timestamp()}.wav"
    file_data = {
        "name": title,
        "type": "audio/x-wav",  # Change the MIME type according to your file type
        "bits": xmlrpc_client.Binary(wav_bytes.getvalue()),
    }
    wp_client = Client(wordpress_url, username, password)
    for _ in range(4):
        try:
            # Upload the file to WordPress Media Library
            media_response = wp_client.call(media.UploadFile(file_data))

            # Handle the response
            if media_response:
                print(
                    "File successfully uploaded to WordPress with attachment ID:",
                    media_response,
                )

                # Create product data for WooCommerce
                product_data = {
                    "status": "pending",
                    "name": title,
                    "type": "simple",
                    "regular_price": "1.00",  # Set the price as needed
                    "sku": str(uuid.uuid4()),
                    "downloadable": True,
                    "download_limit": -1,
                    "download_expiry": -1,
                }

                # Authenticate with WooCommerce API
                wc_api = API(
                    url=woocommerce_url,
                    consumer_key=consumer_key,
                    consumer_secret=consumer_secret,
                    version="wc/v3",
                )

                # Create the product
                response = wc_api.post("products", product_data)

                # Handle the response
                if response.status_code == 201:
                    print(
                        "Product successfully created in WooCommerce:", response.json()
                    )
                    # Update product to add downloadable file URL
                    product_update_data = {
                        "downloads": [
                            {
                                "name": media_response["title"],
                                "file": media_response["link"],
                            }
                        ]
                    }
                    product_id = response.json().get("id")
                    response = wc_api.put(f"products/{product_id}", product_update_data)

                    if response.status_code == 200:
                        print(
                            "Downloadable file URL added to product:", response.json()
                        )
                        return (
                            response.json()["permalink"],
                            response.json()["permalink"].split("p=")[-1],
                        )
                    else:
                        print(
                            "Error adding downloadable file URL to product:",
                            response.text,
                        )
                else:
                    print("Error creating product in WooCommerce:", response.text)
            else:
                print("Error uploading file to WordPress.")
            break
        except Exception as e:
            print("Error:", e)
            continue  # Retry on error

    # If upload fails, return placeholders
    return "https://songlabai.com/contact_us/", "N/A"



# Streamlit app title
st.title("Songlab AI")
#cookies = cookie_manager.get_all(key='init_get_all')


# Ensure session state keys are initialized
if "jwt_token" not in st.session_state:
    st.session_state["jwt_token"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if 'login_needed' not in st.session_state:
    st.session_state['login_needed'] = False
if 'generate_audio_params' not in st.session_state:
    st.session_state['generate_audio_params'] = None


if st.session_state.get("jwt_token"):
    if st.button("Log out", key="logout_button"):
        # Clear all user-specific session state
        st.session_state.clear()
        st.success("Logged out successfully!")
        st.rerun()
        
def get_api_headers():
    return {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
    }


# Function to get the auth headers using the current jwt_token
def get_auth_headers():
    jwt_token = st.session_state.get("jwt_token")
    if jwt_token:
        return {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        }
    else:
        return {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        }

        
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    st.write("Session ID:", st.session_state["session_id"])
    st.write("Current JWT Token:", st.session_state.get("jwt_token"))
    st.write("Session State:", st.session_state)


# Initialize session state variables
if "vocal_audio" not in st_state:
    st_state.vocal_audio = None
if "vocal_sample_rate" not in st_state:
    st_state.vocal_sample_rate = None

if "audio" not in st_state:
    st_state.audio = None
if "audio_pydub" not in st_state:
    st_state.audio_pydub = None
if "audio_sample_rate" not in st_state:
    st_state.audio_sample_rate = None

if "augmented_audio" not in st_state:
    st_state.augmented_audio = None
if "augmented_audio_pydub" not in st_state:
    st_state.augmented_audio_pydub = None
if "augmented_audio_sample_rate" not in st_state:
    st_state.augmented_audio_sample_rate = None


genres = [
    "Pop",
    "Rock",
    "Hip Hop",
    "Jazz",
    "Blues",
    "Country",
    "Classical",
    "Electronic",
    "Reggae",
    "Folk",
    "R&B",
    "Metal",
    "Punk",
    "Indie",
    "Dance",
    "World",
    "Gospel",
    "Soul",
    "Funk",
    "Ambient",
    "Techno",
    "Disco",
    "House",
    "Trance",
    "Dubstep",
]
genre = st.selectbox("Select Genre:", genres)
energy_levels = ["Low", "Medium", "High"]
energy_level = st.radio("Energy Level:", energy_levels, horizontal=True)
description = st.text_input("Description:", "")
tempo = st.slider("Tempo (in bpm):", min_value=40, max_value=100, value=60, step=5)

# First, check subscription status if user is logged in
if st.session_state.get("jwt_token"):
    auth_headers = get_auth_headers()
    subscription_url = "https://songlabai.com/wp-json/custom-api/v1/subscription"
    subscription_response = requests.get(subscription_url, headers=auth_headers)
    
    if subscription_response.status_code == 200:
        subscription_data = subscription_response.json()
        subscription_plan_id = subscription_data.get("subscription_plan_id")
        # Free tier has plan ID 576 or null
        is_free_tier = not subscription_plan_id or subscription_plan_id == 576
    else:
        is_free_tier = True  # Default to free tier if unable to verify
else:
    is_free_tier = True  # Default to free tier if not logged in

# First, check subscription status if user is logged in
if st.session_state.get("jwt_token"):
    auth_headers = get_auth_headers()
    subscription_url = "https://songlabai.com/wp-json/custom-api/v1/subscription"
    subscription_response = requests.get(subscription_url, headers=auth_headers)
    
    if subscription_response.status_code == 200:
        subscription_data = subscription_response.json()
        # Convert subscription_plan_id to string for comparison
        subscription_plan_id = str(subscription_data.get("subscription_plan_id"))
        is_free_tier = subscription_plan_id == "576" or subscription_data.get("status") == "no_subscription"
        
# Show appropriate duration slider based on subscription tier
if is_free_tier:
    duration = st.slider(
        "Duration (in seconds):",
        min_value=15,
        max_value=30,  # Restricted to 30 seconds for free tier
        value=30,
        step=1,
        help="Free tier users are limited to 30-second generations. Upgrade to create longer tracks!"
    )
else:
    duration = st.slider(
        "Duration (in seconds):",
        min_value=15,
        max_value=300,
        value=30,
        step=1
    )


def convert_audio_segment_to_float_array(audio_pydub):
    """
    Convert a pydub AudioSegment to a NumPy array of type float32.

    Args:
    audio_pydub (AudioSegment): The AudioSegment object to be converted.

    Returns:
    np.ndarray: A NumPy array containing the audio data as float32.
    """
    # Get the raw audio data as a sequence of samples
    samples = audio_pydub.get_array_of_samples()

    # Convert the samples to a NumPy array and normalize to float32
    audio_array = np.array(samples).astype(np.float32)

    # Normalize the audio array to range between -1.0 and 1.0
    max_val = 2**15  # Assuming 16-bit audio, modify this if using different bit depths
    audio_array /= max_val
    return audio_array

def time_post_request(api_url, headers=None, payload=None, timeout=None):
    """
    Times the execution of a POST request.

    Parameters:
    - api_url (str): The URL to which the POST request is sent.
    - headers (dict): The headers to include in the POST request.
    - payload (dict): The payload to include in the POST request.
    - timeout (int): The timeout value in seconds for the POST request (optional).

    Returns:
    - response (requests.Response): The response object returned by the POST request.
    - execution_time (float): The time it took to execute the POST request.
    """
    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return response

def check_server_status():
    """Check if the server is warm by making a minimal request"""
    warmup_payload = {"inputs": {"prompt": "warmup", "duration": 1}}
    try:
        response = time_post_request(
            API_URL,
            headers=get_api_headers(),
            payload=warmup_payload,
            timeout=10  # Short timeout for quick check
        )
        return response is not None and response.status_code == 200
    except:
        return False

def wait_for_server_warmup(status_placeholder, progress_placeholder, subscription_info_placeholder, stage_placeholder):
    """Handle server warmup with detailed progress stages and proceed immediately when server is ready"""
    stages = [
        {"time": 0, "message": "🚀 Initializing server resources..."},
        {"time": 60, "message": "⚙️ Loading the AI model into memory..."},
        {"time": 180, "message": "🔧 Optimizing model parameters..."},
        {"time": 300, "message": "📊 Configuring generation settings..."},
        {"time": 360, "message": "✨ Finalizing server preparation..."}
    ]
    
    subscription_info = """
    ### 📋 **Subscription Tiers**

    **Free User**
    - 🎵 **Generations:** 3 music generations
    - 📥 **Downloads:** No downloads allowed
    - 🎼 **Submit your track from dashboard**
    - 💰 **Receive 50% commission**
    - 💵 **Tracks sale for $1 on website**

    **Tier 2 - $24.99/month**
    - 🎵 **Generations:** Up to 3 music generations per day
    - 📥 **Downloads:** 1 download per month
    - 🔒 **Private Tracks:** Create unpublished, private tracks
    - 📊 **Dashboard:** Access your profile and save up to 2 tracks

    **Tier 3 - $99.99/month**
    - 🎵 **Generations:** Up to 5 music generations per day
    - 📥 **Downloads:** 5 downloads per month
    - 📊 **Dashboard:** Full profile access for managing and saving your tracks
    - 💾 **Storage:** Save up to 10 tracks

    **Tier 4 - $269.97/month**
    - 🎵 **Generations:** Up to 20 music generations per day
    - 📥 **Downloads:** 15 downloads per month
    - 🏢 **Commercial Use:** Includes rights for commercial purposes
    - 📊 **Dashboard:** Advanced profile management for commercial accounts

    ### 🚀 **Why Subscribe?**
    Upgrade now to unlock more features and enhance your music creation experience!
    """

    start_time = time.time()
    total_warmup_time = 420  # 7 minutes
    current_stage = 0
    last_check_time = 0
    
    while time.time() - start_time < total_warmup_time:
        elapsed = time.time() - start_time
        remaining = total_warmup_time - elapsed
        progress = elapsed / total_warmup_time
        
        # Update current stage
        while current_stage < len(stages) - 1 and elapsed > stages[current_stage + 1]["time"]:
            current_stage += 1
            
        minutes_remaining = int(remaining // 60)
        seconds_remaining = int(remaining % 60)
        
        # Update UI
        status_placeholder.markdown("### 🔄 AI Server Initialization in Progress")
        progress_placeholder.progress(progress)
        stage_placeholder.info(
            f"{stages[current_stage]['message']}\n\n"
            f"⏳ Estimated time remaining: {minutes_remaining}m {seconds_remaining}s\n\n"
            "ℹ️ **Why the wait?** The AI model needs time to load and prepare when the server is inactive.\n"
            "Upgrading to a premium plan keeps the server active longer, reducing or eliminating wait times!\n"
            "👉 [Upgrade Now](https://songlabai.com/subscribe/)"
        )
        subscription_info_placeholder.markdown(subscription_info)
        
        # Check if server is ready every 5 seconds
        if elapsed - last_check_time >= 5:
            last_check_time = elapsed
            if check_server_status():
                status_placeholder.success("🎉 Server is ready! Proceeding to music generation...")
                return True
                
        time.sleep(0.1)
        
    return False


def generate_audio(genre, energy_level, tempo, description, duration):
    # Create placeholders
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    subscription_placeholder = st.empty()
    stage_placeholder = st.empty()
    
    try:
        # First, verify user is logged in and has permissions
        if st.session_state.get("jwt_token") is None:
            st.error("You must be logged in to generate audio.")
            return None
                
        auth_headers = get_auth_headers()
        
        # Check subscription status
        subscription_url = "https://songlabai.com/wp-json/custom-api/v1/subscription"
        subscription_response = requests.get(subscription_url, headers=auth_headers)
        if subscription_response.status_code != 200:
            st.error("Failed to retrieve your subscription status.")
            return None
                
        subscription_data = subscription_response.json()
        
        # Check for free tier duration restriction
        is_free_tier = subscription_plan_id == "576" or subscription_data.get("status") == "no_subscription"
        if is_free_tier:
            if duration > 30:
                st.warning("⚠️ Free tier users are limited to 30-second generations.")
                st.info("💡 Upgrade to a premium plan to generate longer tracks!\n\n" + 
                       "**Available Plans:**\n" +
                       "- **$24.99/month:** Up to 3 generations per day, 1 download per month\n" +
                       "- **$99.99/month:** Up to 5 generations per day, 5 downloads per month\n" +
                       "- **$269.97/month:** Up to 20 generations per day, 15 downloads per month\n\n" +
                       "👉 [Upgrade Now](https://songlabai.com/subscribe/)")
                duration = 30  # Force duration to 30 seconds for free tier
                
        # Check if server is cold
        if not check_server_status():
            status_placeholder.warning(
                "🔄 **Server is currently starting up.**\n\n"
                "Our AI model needs some time to load and prepare when the server has been inactive.\n"
                "This can take up to **7 minutes** for free users.\n\n"
                "💡 **Upgrade to a premium plan to reduce or eliminate wait times!**\n"
                "👉 [Upgrade Now](https://songlabai.com/subscribe/)"
            )
                
            # Wait for server to warm up with progress display
            server_ready = wait_for_server_warmup(
                status_placeholder,
                progress_placeholder,
                subscription_placeholder,
                stage_placeholder
            )
                
            if not server_ready:
                st.error(
                    "❌ Server initialization timed out.\n\n"
                    "Consider upgrading to premium for reliable, fast access.\n\n"
                    "👉 [Upgrade Now](https://songlabai.com/subscribe/)"
                )
                return None
                    
        # Clear warmup messages
        for placeholder in [status_placeholder, progress_placeholder, subscription_placeholder, stage_placeholder]:
            placeholder.empty()
                
        # Prepare generation request
        prompt = f"Genre: {genre}, Energy Level: {energy_level}, Tempo: {tempo}, Description: {description}"
        payload = {"inputs": {"prompt": prompt, "duration": duration}}
        api_headers = get_api_headers()
        
        # Make the generation request
        with st.spinner("🎵 Generating your music... This may take a few moments."):
            response = time_post_request(API_URL, headers=api_headers, payload=payload, timeout=600)
            
            if response and response.status_code == 200:
                st.success("✨ Music generated successfully!")
                # Pass all parameters to load_and_play_generated_audio
                load_and_play_generated_audio(
                    response=response,
                    genre=genre,
                    energy_level=energy_level,
                    tempo=tempo,
                    description=description,
                    duration=duration
                )
                
                # Update generation count
                update_generation_url = "https://songlabai.com/wp-json/custom-api/v1/update-generation-count"
                requests.post(update_generation_url, headers=auth_headers)
                
                return response  # Return the response for email notification
            else:
                st.error(
                    "❌ Failed to generate audio.\n\n"
                    "This might be due to high server load or an error.\n\n"
                    "💡 **Tip:** Premium users experience fewer failures and faster generation times.\n\n"
                    "👉 [Upgrade Now](https://songlabai.com/subscribe/)"
                )
                return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def download_audio():
    if st_state.audio_pydub is None:
        st.error("No audio available to download.")
        return

    # Convert AudioSegment to bytes
    audio_bytes = st_state.audio_pydub.export(format="wav").read()

    # Create a download button
    st.download_button(
        label="Download Generated Audio",
        data=audio_bytes,
        file_name="generated_audio.wav",
        mime="audio/wav",
        on_click=update_download_count
    )

def notify_admin_of_generation(user_id, genre, energy_level, tempo, description, duration, perm_link, product_code):
    """
    Send a notification email to admin@songlabai.com when a song is generated
    """
    sender_email = "admin@songlabai.com"
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = sender_email
    msg["Subject"] = f"New Song Generated - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    body = f"""
    New Song Generation Details:
    
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    User ID: {user_id}
    
    Song Parameters:
    - Genre: {genre}
    - Energy Level: {energy_level}
    - Tempo: {tempo}
    - Duration: {duration} seconds
    - Description: {description}
    
    Links:
    - Permanent Link: {perm_link}
    - Product Code: {product_code}
    
    --
    Automated notification from SongLab AI
    """
    
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✓ Admin notification email sent successfully")
    except Exception as e:
        print(f"× Failed to send admin notification email: {str(e)}")
        import traceback
        print(traceback.format_exc())

def update_download_count():
    auth_headers = get_auth_headers()
    try:
        notification_url = "https://songlabai.com/wp-json/songlab/v1/notify"
        
        user_id = None
        if st.session_state.get("jwt_token"):
            try:
                decoded = jwt.decode(st.session_state["jwt_token"], options={"verify_signature": False})
                user_id = decoded.get('data', {}).get('user', {}).get('id')
                print(f"Decoded JWT token: {decoded}")  # Debug print
            except Exception as e:
                print(f"Error decoding JWT: {str(e)}")
                user_id = None
        
        payload = {
            "user_id": user_id
        }
        
        print(f"Sending notification to {notification_url}")
        print(f"With payload: {payload}")
        print(f"JWT token: {st.session_state.get('jwt_token')}")
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(notification_url, json=payload, headers=headers)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        try:
            print(f"Response JSON: {response.json()}")
        except:
            print(f"Raw response text: {response.text}")
        
        if response.status_code == 200:
            print("✓ Admin notification sent successfully")
            return True
        else:
            print(f"× Failed to send admin notification (HTTP {response.status_code})")
            return False
            
    except Exception as e:
        print(f"× Error sending notification: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False



def load_and_play_generated_audio(response, genre=None, energy_level=None, tempo=None, description=None, duration=None):
    import soundfile as sf
    from io import BytesIO

    response_eval = json.loads(response.content)
    generated_audio = response_eval[0]["generated_audio"]
    sample_rate = response_eval[0]["sample_rate"]

    # Check if the audio is stereo or mono
    if isinstance(generated_audio[0], list):  # Stereo
        audio_array = np.array(generated_audio).T
    else:  # Mono
        audio_array = np.array(generated_audio)

    # Convert the float32 audio data to int16
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        audio_array = audio_array / max_val  # Normalize to [-1.0, 1.0]
    int_audio = np.int16(audio_array * 32767)

    # Write the audio data to a BytesIO buffer using soundfile
    audio_buffer = BytesIO()
    sf.write(audio_buffer, int_audio, sample_rate, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)

    # Read the audio data into an AudioSegment
    audio_segment = AudioSegment.from_file(audio_buffer, format="wav")

    st_state.audio_pydub = audio_segment
    st_state.audio_sample_rate = sample_rate
    st_state.augmented_audio_pydub = st_state.audio_pydub

    # Play the audio
    st.audio(audio_buffer.getvalue())

    # Save the audio to WordPress and get perm_link and product_code
    perm_link, product_code = save_to_wordpress(audio_array, sample_rate)

    # Get user ID from JWT token
    user_id = None
    if st.session_state.get("jwt_token"):
        try:
            decoded = jwt.decode(st.session_state["jwt_token"], options={"verify_signature": False})
            user_id = decoded.get('user_id', 'Unknown')
        except:
            user_id = 'Unknown'

    # Send email notification using the separate function
    notify_admin_of_generation(
        user_id=user_id,
        genre=genre,
        energy_level=energy_level,
        tempo=tempo,
        description=description,
        duration=duration,
        perm_link=perm_link,
        product_code=product_code
    )

    # Define col_btn and col_text
    col_btn, col_text = st.columns([2, 4])

    with col_btn:
        st.markdown("[Publish your Song](https://songlabai.com/contact_us/)")
        st.markdown("[Download Song](https://songlabai.com/download-music/)")

    with col_text:
        st.write(
            f"To Publish, please contact the admin by sending the following link: {perm_link}"
        )
        st.write(f"To download use the following product code: {product_code}")

if st.button("Generate Audio", key="generate_audio_button"):
    if genre and energy_level and description and tempo:
        if st.session_state.get("jwt_token") is None:
            # User is not logged in
            st.session_state['login_needed'] = True
            st.session_state['generate_audio_params'] = {
                'genre': genre,
                'energy_level': energy_level,
                'tempo': tempo,
                'description': description,
                'duration': duration
            }
            st.warning("Please log in to generate audio.")
        else:
            generate_audio(genre, energy_level, tempo, description, duration)
    else:
        st.info("Description field is required.")

if st.session_state.get('login_needed') and st.session_state.get("jwt_token") is None:
    st.header("Please log in to continue")
    username = st.text_input("Username", key="username_input")
    password = st.text_input("Password", type="password", key="password_input", value="")

    if st.button("Log in", key="login_button"):
        login_url = "https://songlabai.com/wp-json/jwt-auth/v1/token"
        data = {
            "username": username,
            "password": password
        }
        response = requests.post(login_url, data=data)
        if response.status_code == 200 and 'token' in response.json():
            result = response.json()
            st.session_state["jwt_token"] = result["token"]
            st.session_state["session_id"] = str(uuid.uuid4())  # Generate new session ID
            st.success("Logged in successfully!")
            st.session_state['login_needed'] = False
            # Trigger a rerun to immediately hide the login form
            st.rerun()
            # Retrieve the parameters and generate audio if they were stored
            params = st.session_state.get('generate_audio_params')
            if params:
                generate_audio(**params)
                st.session_state['generate_audio_params'] = None
        else:
            st.error("Invalid username or password")
            # Keep login_needed True
    # Add Register link
    st.markdown("Don't have an account? [Register here](https://songlabai.com/register/)")
else:
    pass  # Do nothing if login is not needed


# Post-processing options
st.header("Post-processing Options")

vocal_file = st.file_uploader(
    "Upload Vocal File", type=["mp3", "wav", "ogg", "flac", "aac"]
)
if vocal_file:
    st_state.vocal_audio = vocal_file.read()
    # st.audio(st_state.vocal_audio, format="audio/wav")

# Mixing
mix_vocals = st.checkbox("Mix Vocals")

if mix_vocals and st_state.vocal_audio is not None:
    with NamedTemporaryFile() as f:
        f.write(st_state.vocal_audio)
        st_state.vocal_audio = AudioSegment.from_file(f.name)
    st_state.augmented_audio_pydub = st_state.augmented_audio_pydub.overlay(
        st_state.vocal_audio, position=100
    )
    # st.audio(st_state.augmented_audio_pydub.export().read())
    st_state.augmented_audio = convert_audio_segment_to_float_array(
        st_state.augmented_audio_pydub
    )
    st_state.augmented_audio_sample_rate = st_state.augmented_audio_pydub.frame_rate
elif not mix_vocals and st_state.vocal_audio is not None:
    st_state.augmented_audio_pydub = st_state.audio_pydub
    st_state.augmented_audio_sample_rate = st_state.audio_pydub.frame_rate
# Mastering
st.header("Mastering")
st.markdown("")
# Volume Balance, Compression Ratio, and Reverb Amount
vol_col, pitch_shift,buttons_col = st.columns([2, 2, 2.5,])
with buttons_col:
    with st.container(height=371, border=True):
        st.markdown("")
        apply_stereo = st.button("Apply Stereo Effect")
        st.markdown("")
        reverse = st.button("Apply Audio Reverse ")
        st.markdown("")
        reset_post_processing = st.button("Undo All Post-processings")
        st.markdown("")

with vol_col:
    with st.container(border=True):
        volume_balance = svs.vertical_slider(
            "Volume Balance",
            min_value=-10.0,
            max_value=10.0,
            default_value=0.0,
            step=0.1,
            slider_color="green",
            track_color="lightgray",
            thumb_color="red",
            thumb_shape="pill",
        )
        vol_button = st.button("Apply Vol-Balance")


# Pitch shifting
with pitch_shift:
    with st.container(border=True):
        pitch_semitones = svs.vertical_slider(
            label="Pitch (semitones)",
            min_value=-12,
            max_value=12,
            default_value=0,
            step=1,
            slider_color="red",
            track_color="lightgray",
            thumb_color="red",
            thumb_shape="pill",
        )
        pitch_shift_button = st.button(
            "Apply Pitch Shift",
        )


if st_state.augmented_audio_pydub is not None:
    if vol_button:
        st_state.augmented_audio_pydub = st_state.augmented_audio_pydub + volume_balance

    if apply_stereo:
        st_state.augmented_audio_pydub = st_state.augmented_audio_pydub.pan(
            -0.5
        ).overlay(st_state.augmented_audio_pydub.pan(0.5))
    if reverse:
        st_state.augmented_audio_pydub = st_state.augmented_audio_pydub.reverse()
    if pitch_shift_button:
        st_state.augmented_audio_pydub = st_state.augmented_audio_pydub._spawn(
            st_state.augmented_audio_pydub.raw_data,
            overrides={
                "frame_rate": int(
                    st_state.augmented_audio_pydub.frame_rate
                    * (2 ** (pitch_semitones / 12.0))
                )
            },
        )

# Display the final audio
if st_state.augmented_audio_pydub is not None:
    st.audio(st_state.augmented_audio_pydub.export().read())
    # sample_rate = st_state.augmented_audio_sample_rate
    # st.audio(st_state.augmented_audio, format="audio/wav", sample_rate=sample_rate*2, start_time=0)

st.link_button(
    label="⬇️ Download/Save",
    url="https://songlabai.com/subscribe/",
    type="primary",
    use_container_width=True,
)
# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
# # color: rgb(204, 49, 49);
# # background-color: ;
# # border-radius: 5%;
# backgroud-color: #00ff00;
# }
# # </style>""", unsafe_allow_html=True)

if reset_post_processing and st_state.audio_pydub is not None:
    st_state.augmented_audio_pydub = st_state.audio_pydub