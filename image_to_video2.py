import torch
from diffusers import StableVideoDiffusionPipeline
import gradio as gr
from PIL import Image
import tempfile
import imageio
import os
import subprocess

# ⚙️ Carrega o modelo SVD
model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
pipe.enable_model_cpu_offload()

# 🧠 Upscale usando Real-ESRGAN
def upscale_frame_pil(pil_image):
    input_path = "input_temp.png"
    output_path = "results/temp_out.png"

    pil_image.save(input_path)
    subprocess.run([
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4",
        "-i", input_path,
        "--outscale", "2"
    ])
    return Image.open(output_path)

# 🚀 Geração de vídeo
def gerar_video(image, prompt, duration, fps, criatividade):
    if image is None:
        return None
    
    num_frames = min(int(duration * fps), 25)

    # Mantém proporção e redimensiona para 512x512 (modelo exige)
    img_w, img_h = image.size
    aspect_ratio = img_w / img_h

    if aspect_ratio > 1:
        new_h = 512
        new_w = int(aspect_ratio * new_h)
    else:
        new_w = 512
        new_h = int(new_w / aspect_ratio)

    image = image.convert("RGB").resize((new_w, new_h))
    image = image.crop(((new_w - 512) // 2, (new_h - 512) // 2, (new_w + 512) // 2, (new_h + 512) // 2))

    # Semente (criatividade)
    torch.manual_seed(int(100 - criatividade * 100))

    # Aplica o pipeline
    video_frames = pipe(image, decode_chunk_size=8, num_frames=num_frames).frames[0]

    # Faz upscale dos frames
    upscaled_frames = [upscale_frame_pil(frame) for frame in video_frames]

    # Salva vídeo
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
        imageio.mimsave(video_path, upscaled_frames, fps=fps)

    return video_path

# 🎛 Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Stable Video Diffusion + Real-ESRGAN (Offline)")

    with gr.Row():
        image_input = gr.Image(type="pil", label="📸 Imagem Base")
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (não usado no modelo ainda)", placeholder="ex: estilo Ghibli")
            duration = gr.Slider(1, 5, value=5, step=1, label="⏱ Duração (segundos)")
            fps = gr.Slider(4, 12, value=12, step=1, label="🎞 FPS")
            criatividade = gr.Slider(0, 1, value=0.5, step=0.1, label="🌈 Criatividade")

    gerar_btn = gr.Button("🚀 Gerar Vídeo")
    video_output = gr.Video(label="Resultado")

    gerar_btn.click(fn=gerar_video, inputs=[image_input, prompt, duration, fps, criatividade], outputs=video_output)

demo.launch(debug=True, share=True)
