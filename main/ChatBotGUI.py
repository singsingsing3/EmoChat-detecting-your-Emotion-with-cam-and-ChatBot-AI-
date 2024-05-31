import tkinter as tk
import cv2 as cv

class SimpleChatBotGUI:
    def __init__(self, chatbot, master):
        self.chatbot = chatbot
        self.master = master
        self.master.title("EmoChat")
        
        self.label = tk.Label(master, text="Please Enjoy EmoChat!")
        self.label.pack()

        self.text_dialog = tk.Text(master)
        self.text_dialog.pack()

        self.label = tk.Label(master, text="Your message:")
        self.label.pack()
        
        self.entry_msg = tk.Entry(master)
        self.entry_msg.pack()
        
        self.button_send = tk.Button(master, text="Send Your Message", command=self.handle_button)
        self.button_send.pack()

        # 초기 메시지 표시
        initial_emotion, initial_confidence = self.chatbot.emotion_detector.predict_emotion()
        # 사용 설명서 표시
        self.text_dialog.insert(tk.END, f"<<사용 설명서>>\n"
                                f"안녕하세요 당신의 감정을 읽으며 채팅하는 사랑스러운 프로그램입니다.\n"
                                
                                f"'emotion'이라고 입력하시면 당신의 표정을 읽고 다시 분석해 드릴게요\n"
                                f"'exit'이라고 입력하시면 프로그램이 종료됩니다.\n"
                                f"저랑 편하게 대화를 시작해봐요!\n\n"
                                f"<<<Usage Manual>>\n"
                                f"Hello I'm a lovely program of chatting while reading your feelings.\n"
                                f"If you enter 'emotion' and I'll read your expression and analyze it again.\n"
                                f"If you enter 'exit' to exit the program.\n"
                                f"Let's start talking comfortably with me!\n\n")
        
        self.text_dialog.insert(tk.END, f"Emochat: 당신은 지금 {initial_emotion} ({initial_confidence:.2f}%) 해보이네요. 오늘 무슨일이 있었나요?\n"
                                        f"Emochat: You look {initial_emotion} ({initial_confidence:.2f}%) now. What happened today?\n\n")
        
        # OpenCV 카메라 프레임 표시
        self.update_camera()

    def handle_button(self):
        msg = self.entry_msg.get()
        self.text_dialog.insert(tk.END, "You: " + msg + "\n")
        response = self.chatbot.reply(msg)
        self.text_dialog.insert(tk.END, "EmoChat: " + response + "\n\n")
        self.entry_msg.delete(0, tk.END)
        if msg.lower() == 'exit':  # 유저가 프로그램 종료를 원하는 경우
            self.master.quit()
            cv.destroyAllWindows()

    def update_camera(self):
        frame = self.chatbot.emotion_detector.get_frame()
        if frame is not None:
            cv.imshow('Emotion_Predict', frame)
        self.master.after(10, self.update_camera)

        
