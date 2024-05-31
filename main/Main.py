import tkinter as tk
from ChatBot import ChatBot
from EmotionDetect import EmotionDetector
from ChatBotGUI import SimpleChatBotGUI

#ChatBot.py에서 당신의 OPENAI API KEY를 입력하세요
#In Chatbot.py, enter your OPENAI API KEY
if __name__ == "__main__":
    emotion_detector = EmotionDetector()
    chatbot = ChatBot(emotion_detector)
    root = tk.Tk()
    app = SimpleChatBotGUI(chatbot, root)
    root.mainloop()
