import gradio as gr # this will help us write python code for web interface development.
import base64
import io
import os
from PIL import Image

# Import for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI # this will help us to work with google gemini model
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser



def encode_image_to_base64(image:Image.Image) -> str:
  buffer = io.BytesIO()                                  #this creates a buffer memory in RAM
  image.save(buffer, format = "PNG")
  return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Initialize Gemini LLM with 'gemini-2.5-flash' for multimodal capabilities
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chain = llm | StrOutputParser()

scope_text = """### What you'll get after analysis:
**🍱 Items Detected**
A table listing every food item on your plate with estimated quantity, calories, protein, carbs, and fat.
**🔢 Total Nutrition**
Summed-up macros — total calories, protein, carbs, and fat for the entire meal.
**📊 Nutritional Score**
A quick score (out of 10) rating how balanced your meal is across protein, carbs, and fats.
**➕ What's Missing**
3 personalised suggestions for foods or nutrients to add for a more complete meal based on your diet and meal type.
---
*Upload your plate image and click **Analyze** to begin.*
"""

def analyze_plate(image: Image.Image, meal_type:str, diet_type:str):
  if image is None:
    return "No Image Uploaded"
  image_base64 = encode_image_to_base64(image)
  prompt = f"""You are a nutritionist. Analyze this {meal_type} plate for a {diet_type} diet.
  Respond in this exact format:
  ## Items Detected
  | Food Item | Quantity | Calories | Protein (g) | Carbs (g) | Fat (g) |
  |-----------|----------|----------|-------------|-----------|---------|
  (one row per item)
  ## Total Nutrition
  - Calories: X kcal
  - Protein: X g | Carbs: X g | Fat: X g
  ## Nutritional Score
  Rate out of 10: Protein X/10 | Carbs X/10 | Fats X/10 | Overall X/10
  ## What's Missing
  List 3 foods/nutrients to add for a balanced {meal_type} ({diet_type}), with a brief reason each."""

  message = HumanMessage(content = [
                        {"type":"image_url", "image_url":{"url":f"data:image/png;base64, {image_base64}"}},
                          {"type":"text", "text":prompt}])

  result = ""
  for chunk in llm.stream([message]):
    result = result + chunk.content
    yield gr.update(value = result), gr.update(interactive = False)

  yield gr.update(value = result), gr.update(interactive = False)


def set_analysis():
  return gr.update(value = "Analyzing your plate.....please wait."), gr.update(interactive = False)

def clear_all():
  return None, gr.update(value =scope_text), gr.update(interactive = True)

  # interface
with gr.Blocks(title = "Plate Calorie Analyzer") as demo:
  gr.Markdown("# 🥗 Plate Calorie Analyzer")

  with gr.Row():
    with gr.Column(scale = 1):
      image_input = gr.Image(type = "pil", label = "Upload Image") # Removed comma
      meal_type = gr.Radio(["Breakfast", "Lunch", "Dinner"], value = "Lunch", label = "Meal Type") # Removed comma
      diet_type = gr.Radio(["Vegan", "Vegetarian", "Non-Vegetarian"], value = "Vegetarian", label = "Diet Type") # Removed comma
      with gr.Row():
        submit_btn = gr.Button("Analyze")
        clear_btn = gr.Button("Clear")

    with gr.Column(scale = 1):
      output = gr.Markdown(value =scope_text , label = "Analysis")


  submit_btn.click(fn = set_analysis,
                  inputs = [],
                  outputs = [output, submit_btn],
                  queue = False).then(
                      fn = analyze_plate,
                      inputs = [image_input, meal_type, diet_type],
                      outputs = [output, submit_btn]
                  )


  clear_btn.click(fn = clear_all,
                inputs = [],
                outputs = [image_input, output,submit_btn ])
  demo.launch(share = True, show_error = True)