import streamlit as st
import transformers
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import requests
from PIL import Image
import base64

# Convert an image to text using an image captioning model
def img2text(image):
    pipe = transformers.pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = pipe(image)[0]["generated_text"]
    return text

# Generate a story from a scenario or input text
def generate_story(scenario, llm):
    template = """You are a story teller. You get a scenario as an input text, and generates a short story out of it.
    Context: {scenario}
    Story:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    chain = LLMChain(prompt=prompt, llm=llm)
    full_output = chain.predict(scenario=scenario)
    story_start = full_output.find("Story:")
    if story_start != -1:
        story = full_output[story_start + 6:].strip()  # +6 to skip "Story:"
    else:
        story = full_output.strip()
    
    return story

# Convert text to speech
def text2speech(text, api_token):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        st.error(f"Error in text-to-speech API: {response.content}")
        return None
    return response.content

# Text enhancement function (rewriting sentences like in a novel)
def enhance_text(input_text, llm):
    template = """You are a professional novelist. Rewrite the following text to make it more vivid, descriptive, and novel-like.
    Text: {input_text}
    Enhanced Text:
    """
    prompt = PromptTemplate(template=template, input_variables=["input_text"])
    chain = LLMChain(prompt=prompt, llm=llm)

    enhanced_text = chain.predict(input_text=input_text)
    enhance_start = enhanced_text.find("Enhanced Text:")
    if enhance_start != -1:
        enhance = enhanced_text[enhance_start + 14:].strip()  # +13 to skip 
    else:
        enhance = enhanced_text.strip()
    
    return enhance

# Describe an image like in a novel
def describe_image(image, llm):
    # Generate multiple detailed captions using beam search and return sequences
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Generate multiple captions to capture different aspects of the image
    captions = pipe(image)[0]['generated_text']
       
    
    # Use the LLM to expand the combined caption into a detailed, novel-like description
    template = """You are a novelist. Given the following image description, rewrite it into a short novel-like description.
    Basic Description: {captions}
    Novel-like Description:
    """
    prompt = PromptTemplate(template=template, input_variables=["captions"])
    chain = LLMChain(prompt=prompt, llm=llm)

    novel_description = chain.predict(captions=captions)
    descript_start = novel_description.find("Novel-like Description:")
    if descript_start != -1:
        description = novel_description[descript_start + 23:].strip()  # +12 to skip "Description:"
    else:
        description = captions.strip()
    
    return description
    

# Streamlit app
def main():
    st.set_page_config(page_title="TaleCraft", page_icon="✒️")

    st.title("✒️ TaleCraft")
    st.markdown("## Your Creative Writing Assistant")

    with st.expander("💡About This Application"):
        st.markdown("""
        **Created by - Tribuvana Kartikeya G (tribuvan.g@gmail.com)**

This application serves as a creative assistant for novelists and content writers, 
                    offering tools to help transform images and text into compelling narratives. 
                    Whether it's generating stories from visuals, describing scenes with vivid detail, or enhancing your writing, this app provides versatile features to support your novel-writing journey.
                     Explore the options in the sidebar to begin crafting your next masterpiece.
    """)

    # Sidebar options
    st.sidebar.title("Choose an Option")
    option = st.sidebar.selectbox("Select a function", ("Story Generation", "Image Description", "Text Enhancement"))

    # API Token input
    api_token = st.text_input("Enter your Hugging Face API Token:", type="password")
    
    if not api_token:
        st.warning("Please enter your Hugging Face API Token to proceed.")
        return

    # Hugging Face Hub model setup
    repo_id = "tiiuae/falcon-7b-instruct"
    try:
        llm = HuggingFaceHub(huggingfacehub_api_token=api_token,
                             repo_id=repo_id,
                             verbose=False,
                             model_kwargs={"temperature":0.5, "max_new_tokens":500})  # Adjust tokens and temperature as needed
    except ValueError as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check if your API token is correct.")
        return

    if option == "Story Generation":
        st.header("Story Generation")

        # Option to either input text or use an image
        input_choice = st.selectbox("Choose input type:", ["Text Input", "Image Upload"])
        
        if input_choice == "Text Input":
            input_text = st.text_area("Enter the scenario or text:")
            if st.button('Generate Story'):
                if input_text:
                    story = generate_story(input_text, llm)
                    st.write("Generated Story:", story)
                    audio_bytes = text2speech(story, api_token)
                    if audio_bytes:
                        # Save audio to a file
                        with open("temp_audio.wav", "wb") as f:
                            f.write(audio_bytes)
                        
                        # Read the file and encode to base64
                        with open("temp_audio.wav", "rb") as f:
                            audio_base64 = base64.b64encode(f.read()).decode()
                        
                        # Display audio player
                        st.markdown(f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>', unsafe_allow_html=True)
                        
                        # Provide download link
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name="story_audio.wav",
                            mime="audio/wav"
                        )
                else:
                    st.warning("Please enter some text.")
        elif input_choice == "Image Upload":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                if st.button('Generate Story'):
                    # Image to text
                    scenario = img2text(image)
                    st.write("Image Caption:", scenario)

                    # Generate story
                    story = generate_story(scenario, llm)
                    st.write("Generated Story:", story)
                    audio_bytes = text2speech(story, api_token)
                    if audio_bytes:
                        # Save audio to a file
                        with open("temp_audio.wav", "wb") as f:
                            f.write(audio_bytes)
                        
                        # Read the file and encode to base64
                        with open("temp_audio.wav", "rb") as f:
                            audio_base64 = base64.b64encode(f.read()).decode()
                        
                        # Display audio player
                        st.markdown(f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>', unsafe_allow_html=True)
                        
                        # Provide download link
                        st.download_button(
                            label="Download Audio",
                            data=audio_bytes,
                            file_name="story_audio.wav",
                            mime="audio/wav"
                        )

    elif option == "Image Description":
        st.header("Image Description")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button('Describe Image'):
                description = describe_image(image,llm)
                st.write(description)

    elif option == "Text Enhancement":
        st.header("Text Enhancement")
        input_text = st.text_area("Enter the text you want to enhance:")

        if st.button("Enhance Text"):
            enhanced_text = enhance_text(input_text, llm)
            st.write(enhanced_text)

if __name__ == "__main__":
    main()
