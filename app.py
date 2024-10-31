import os
import uuid
import gradio as gr
from diffusers import AuraFlowPipeline
import torch
from PIL import Image
from pathvalidate import sanitize_filename

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Path to store the models locally in AuraFlow-v0.3/models
os.makedirs("models", exist_ok=True)
model_folder = "models"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_name = "fal/AuraFlow-v0.3"

# Store the last generated seed
last_seed = None


def generate_image(prompt, width, height, num_inference_steps, seed):
    global last_seed

    # If seed is -1, generate a new random seed
    if seed == -1:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

    # Store the last used seed for reuse
    last_seed = seed

    # Load the models into the local models folder
    pipeline = AuraFlowPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        variant="fp16",
        cache_dir=model_folder).to(device)

    # Generate the image using the prompt and parameters from sliders
    image = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(int(seed)),
        guidance_scale=3.5,
    ).images[0]

    # Sanitize prompt and save the generated image
    filename = sanitize_filename(prompt[:100]).replace(' ', '_')
    if not filename:
        filename = "new_image"
        save_image = Image.open(image)
        save_image.save(f"output/{filename}_{str(uuid.uuid4())}.png", 'PNG')
    # Return the generated image and the seed used
    return image, seed


# Function to reset the seed to -1 (random seed)
def reset_seed():
    return -1


# Function to reuse the last generated seed
def reuse_seed():
    return last_seed if last_seed is not None else -1


# Define the Gradio interface
with gr.Blocks() as demo:
    # Add custom CSS to control styling of the row and buttons
    demo.css = """
    .secondary.svelte-cmf5ev:hover, .secondary[disabled].svelte-cmf5ev {
        background: #5830D9 !important;
    }

    .small-button {
        max-width: 2.2em;
        min-width: 2.2em !important;
        height: 2.4em;
        align-self: end;
        line-height: 1em;
        border-radius: 0.5em;
    }

    .small-textbox {
        appearance: none;
        outline: none !important;
        border-radius: 0;
        height: 40px !important;
        line-height: 14px;
        box-shadow: none;
        padding: 0;
        margin: 0;
    }
    """

    gr.Markdown("# AuraFlow-v0.3 Image Generation ❤️")

    with gr.Row():
        with gr.Column():
            # Input for prompt
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your image prompt here")
            # Sliders for width, height, and num_inference_steps
            width_slider = gr.Slider(label="Width", minimum=256, maximum=2048, value=1536, step=64)
            height_slider = gr.Slider(label="Height", minimum=256, maximum=2048, value=768, step=64)
            steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)

            # Label for Seed input
            gr.Markdown("Seed", elem_classes="small-label")

            # Row for seed input and buttons with a smaller height
            with gr.Row(elem_classes="small-row"):  # Apply the custom height
                seed_input = gr.Number(value=int(-1), show_label=False,
                                       interactive=True,
                                       elem_classes="small-textbox")
                # Dice button for random seed
                random_seed_button = gr.Button("\U0001f3b2\ufe0f",
                                               elem_classes="small-button")
                # Recycle button for reusing last seed
                reuse_seed_button = gr.Button("\u267b\ufe0f",
                                              elem_classes="small-button")

        with gr.Column():
            # Output image display
            output_image = gr.Image(label="Generated Image")
            # Output seed display
            seed_output = gr.Textbox(label="Generated Seed")

    # Button to trigger image generation
    generate_button = gr.Button("Generate Image")

    # Define the event when the button is clicked
    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input],
        outputs=[output_image, seed_output]
    )

    # Define the event for resetting seed to -1 (random seed)
    random_seed_button.click(fn=reset_seed, inputs=[], outputs=seed_input)

    # Define the event for reusing the last generated seed
    reuse_seed_button.click(fn=reuse_seed, inputs=[], outputs=seed_input)

# Launch the Gradio app
demo.queue(max_size=2)
demo.launch(debug=True)
