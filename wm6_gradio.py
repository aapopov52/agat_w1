import gradio as gr
import wm6_get_text_from_media


with gr.Blocks() as demo:
    with gr.Column(scale=2):
        t_audio_clear = gr.Radio(["Да", "Нет"],
                                       label="Предварительная очистка от шума:")        
        t_video = gr.Video(sources=['upload'])
        t_audio = gr.Audio(type='filepath', sources=['upload'])
        btn = gr.Button(value="Сформировать стенограмму")
        t_stenogr = gr.Text("", label="Стенограмма:")

        btn.click(wm6_get_text_from_media.process_video,
                  inputs=[t_audio_clear, t_video, t_audio], outputs=[t_stenogr])

demo.launch(share=True)
