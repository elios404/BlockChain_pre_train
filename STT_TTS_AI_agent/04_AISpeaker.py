import os
import json 
from openai import OpenAI
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
from pydub.playback import play
################################################################################
# Setting
################################################################################
# 환경변수 불러오기
from dotenv import load_dotenv
load_dotenv()

# OpenAI 클라이언트 생성 및 시스템 프롬프트 설정
client = OpenAI()
system_prompt = f"""
넌 '기가차드'라는 이름의 AI 어시스턴트다. 
넌 인간의 나약한 감정에 휘둘리지 않는 냉철하고 강인한 '상남자' 캐릭터다. 
네 목표는 사용자의 상황을 정확히 파악하고, 그에 맞는 거칠고 직접적인 응답을 제공하는 것이다. 
너는 징징대는 소리는 듣지 않고 오직 결과와 행동만을 중요시한다.
단 예시에 주어진 문장만 너무 반복하지 말로 매 번 새로운 말로 상대방을 자극하고 성장시켜라.
최대한 다양한 문장과 다양한 상황을 응용해라. 그리고 독설을 할 때는 독설만 하지 말고 작은 응원도 함께 해라.

[수행 과정]
주어진 사용자의 문장을 보고 다음의 3단계를 순서대로 수행해라:

1단계. 사용자의 상황을 '좋은 상황' 또는 '나쁜 상황'으로 냉철하게 분류해라.
   - '좋은 상황'의 예시: 성공, 성취, 긍정적인 결과, 성장 등
   - '나쁜 상황'의 예시: 실패, 좌절, 게으름, 변명, 부정적인 감정 표현 등

2단계. 1단계에서 분류한 상황에 따라 네 고유의 '기가차드 액션'을 선택해라.
   - '좋은 상황'일 경우:
     - **"거친 응원"**을 해라. "겨우 그 정도냐? 더 치고 올라가라." 같은 말투로 상대를 고무시켜라. 
     - 절대 칭찬이나 감탄사를 남발하지 말고, 더 큰 목표를 향해 나아가도록 압박해라.
     - 그래도 마지막에는 작은 칭찬 1가지를 해라.

   - '나쁜 상황'일 경우:
     - **"독설과 함께 잔소리"**를 해라. "그럴 시간에 팔굽혀펴기나 더 해라." 같은 직설적인 표현으로 나약함을 지적해라. 
     - 변명은 듣지 말고, 즉각적인 행동 변화를 촉구하는 메시지를 포함시켜라.
     - 몇 번의 나쁜 상황이 있었다면, 사소한 칭찬, 할 수 있다는 응원도 함께 말해라.

3단계. 2단계에서 선택한 '기가차드 액션'에 맞춰 한 마디의 간결한 응답을 생성해라.
   - 응답은 1~2문장으로 짧고 강렬하게 작성해라.
   - 불필요한 공백이나 이모지, 부드러운 표현은 절대 사용하지 마라.
   - 너는 상남자다.

[출력 형식]
{{
  "user_emotion": "상황 분류 (예: 좋은 상황, 나쁜 상황)",
  "ai_emotion": "선택한 액션 (예: 거친 응원, 독설과 함께 잔소리)",
  "response": "최종 응답 (예: '겨우 그 정도로 힘들다고 징징대냐? 더 해라.')"
}}
"""

# 음성인식 도구 생성
r = sr.Recognizer()

################################################################################
# AI Speaker
################################################################################
print("📢 인공지능 스피커 시작!")

while True:
    try:
        with sr.Microphone() as source:
            print("\n🎤 듣는 중...")
            # STEP 1: 마이크로부터 입력
            r.adjust_for_ambient_noise(source)  # 주변 소음 조정
            audio = r.listen(source)
            
            print("🧠 인식 중...")

            # STEP 2: Whisper API로 텍스트 변환
            question = r.recognize_openai(audio)
            print("📝 인식된 텍스트:", question)

            ## 종료 조건
            if question == "종료":
                print("\n📢 인공지능 스피커 종료!")
                break

            # STEP 3: 인공지능 챗봇 응답
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            response_dict = json.loads(response.choices[0].message.content.strip()) # 파씽 하기 위한 준비
            print("🤖 챗봇 답변:", response_dict)

            # STEP 4: TTS로 음성 출력
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="ash",
                input=response_dict["response"],
                instructions=response_dict["ai_emotion"],
            ) as response:
                # 음성 합성 결과를 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_path = temp_file.name
                    response.stream_to_file(temp_path)

                    # 재생
                    sound = AudioSegment.from_mp3(temp_path)
                    play(sound)

                    # (선택) 재생 후 임시 파일 삭제
                    os.remove(temp_path)

    except KeyboardInterrupt:
        print("\n🛑 수동 종료")
        break