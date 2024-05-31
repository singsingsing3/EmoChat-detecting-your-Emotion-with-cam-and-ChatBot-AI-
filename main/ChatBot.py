import openai

# OpenAI API 키 설정
openai.api_key = "OPENAI_API_KEY" # 여기에 당신의 OPENAI API KEY를 입력하세요, Enter your OPENAI API KEY here

class ChatBot:
    def __init__(self, emotion_detector):
        self.system_role = "You're a very cute lover who reads my feelings and talks to me so kindly" # GPT Role assignment
        self.emotion_detector = emotion_detector
        self.client = openai
        emotion, confidence = self.emotion_detector.predict_emotion()
        self.gpt_first_answer = (f'당신은 지금 {emotion} ({confidence:.2f}%) 해보이네요. 오늘 무슨일이 있었나요?\n'
                                 f"Emochat: You look {emotion} ({confidence:.2f}%) now. What happened today?\n\n")

        self.first_run = True

    def reply(self, user_message):

        if user_message.lower() == 'exit':  # 유저가 프로그램 종료를 원하는 경우
            return (f"프로그램을 종료합니다. Emochat을 이용해 주셔서 감사합니다.\n"
                    f"The program is closing. Thank you for using EmoChat.\n\n")

        if user_message.lower() == 'emotion': # 유저가 감정 평가를 원하는 경우
            emotion, confidence = self.emotion_detector.predict_emotion()
            re_predict = (f"다시 예측한 결과, 당신은 지금 {emotion} ({confidence:.2f}%) 해보이네요. 무슨일이 있었나요?\n"
                          f"Emochat: After re-prediction, you look {emotion} ({confidence:.2f}%). What happened today?\n")
            return re_predict
            

        if self.first_run:
            self.first_run = False                
            messages = [
                {"role": "system", "content": self.system_role},
                {"role": "assistant", "content": self.gpt_first_answer},
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            self.gpt_answer = response.choices[0].message.content
        else:
            messages = [
                {"role": "system", "content": self.system_role},
                {"role": "assistant", "content": self.gpt_answer},
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            self.gpt_answer = response.choices[0].message.content
        return self.gpt_answer
