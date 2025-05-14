import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
import re
from functools import lru_cache
import time
import concurrent.futures

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Smart Food Analyzer", layout="wide", page_icon="🍏")

# ------------------- CONFIG ------------------- #
genai.configure(api_key="AIzaSyA0MVpJdhxriWKiOo4pIkeU7fr6iVWADwkof ")
gemini_model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
DAILY_CALORIES = 2000
DAILY_MACROS = {"protein": 50, "carbs": 275, "fats": 70}

# Common food database (Western and Indian)
FOOD_DATABASE = {
    "Chicken Biryani": {"calories": 350, "protein": 15, "carbs": 45, "fats": 12},
    "Paneer Tikka": {"calories": 280, "protein": 18, "carbs": 10, "fats": 20},
    "Dal Tadka": {"calories": 200, "protein": 10, "carbs": 30, "fats": 5},
    "Masala Dosa": {"calories": 320, "protein": 6, "carbs": 50, "fats": 10},
    "Cheeseburger": {"calories": 550, "protein": 25, "carbs": 40, "fats": 30},
    "Caesar Salad": {"calories": 350, "protein": 12, "carbs": 20, "fats": 25},
    "Margherita Pizza": {"calories": 850, "protein": 35, "carbs": 100, "fats": 30},
    "Grilled Salmon": {"calories": 400, "protein": 35, "carbs": 0, "fats": 28},
    "Vegetable Stir Fry": {"calories": 250, "protein": 8, "carbs": 30, "fats": 12}
}

if "current_food_data" not in st.session_state:
    st.session_state.current_food_data = None

# ------------------- CACHED FUNCTIONS ------------------- #
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_gemini_response(_model, prompt, image=None):
    """Cached function for Gemini API calls"""
    if image:
        return _model.generate_content([prompt, image])
    return _model.generate_content(prompt)

@st.cache_data
def process_macros(response_text):
    """Cached macro processing"""
    macros = {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}
    patterns = {
        "calories": r"\*\*Calories\*\*:\s*(\d+)\s*kcal",
        "protein": r"\*\*Protein\*\*:\s*(\d+)\s*g",
        "carbs": r"\*\*Carbs\*\*:\s*(\d+)\s*g",
        "fats": r"\*\*Fats\*\*:\s*(\d+)\s*g"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            macros[key] = int(match.group(1))
    
    vitamins_match = re.search(r"\*\*Notable Vitamins/Minerals\*\*:\s*([a-zA-Z, ]+)", response_text)
    vitamins = vitamins_match.group(1) if vitamins_match else "None"
    
    name_match = re.search(r"\*\*Food Name\*\*:\s*([^\n]+)", response_text)
    food_name = name_match.group(1).strip() if name_match else "Your Food"
    
    return macros, food_name, vitamins

# ------------------- OPTIMIZED IMAGE HANDLING ------------------- #
def process_uploaded_image(uploaded_file):
    """Optimized image processing with tempfile"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name
    return Image.open(temp_path), temp_path

# ------------------- PARALLEL PROCESSING ------------------- #
def analyze_food_parallel(model, image, portion):
    """Run parallelizable operations concurrently"""
    def get_nutrition():
        nutrition_prompt = (
            f"You are a nutritionist AI. The user uploaded a food image and ate about {portion}. "
            "Estimate nutritional values **without giving any ranges**. Return only the following in bullet points: "
            "**Calories** (kcal), **Protein** (g), **Carbs** (g), **Fats** (g), and **Notable Vitamins/Minerals**. "
            "Also suggest a name for this food item in this format: **Food Name**: [your suggestion]"
        )
        return get_gemini_response(model, nutrition_prompt, image)
    
    def get_health_tips():
        return "Health tips placeholder"  # Will be updated after nutrition analysis
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        nutrition_future = executor.submit(get_nutrition)
        health_future = executor.submit(get_health_tips)
        
        return nutrition_future.result(), health_future.result()

# ------------------- VISUALIZATION FUNCTIONS ------------------- #
def plot_macro_comparison(user_macros, food_name="Your Food"):
    foods = list(FOOD_DATABASE.keys())
    proteins = [FOOD_DATABASE[food]["protein"] for food in foods]
    carbs = [FOOD_DATABASE[food]["carbs"] for food in foods]
    fats = [FOOD_DATABASE[food]["fats"] for food in foods]
    
    foods.insert(0, food_name)
    proteins.insert(0, user_macros["protein"])
    carbs.insert(0, user_macros["carbs"])
    fats.insert(0, user_macros["fats"])
    
    df = pd.DataFrame({
        "Food": foods,
        "Protein": proteins,
        "Carbs": carbs,
        "Fats": fats
    })
    
    df_melted = df.melt(id_vars="Food", var_name="Macro", value_name="Value")
    
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_melted, x="Value", y="Food", hue="Macro", 
                    palette={"Protein": "#4ECDC4", "Carbs": "#45B7D1", "Fats": "#FFC154"})
    
    for i, bar in enumerate(ax.patches):
        if i < 3:
            bar.set_edgecolor("#FF0000")
            bar.set_linewidth(2)
            bar.set_alpha(0.9)
    
    plt.title(f"Macronutrient Comparison: {food_name} vs Common Foods", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Amount (g)", fontsize=12)
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Macronutrients")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_line_comparison(user_macros, food_name="Your Food"):
    foods = list(FOOD_DATABASE.keys())
    proteins = [FOOD_DATABASE[food]["protein"] for food in foods]
    carbs = [FOOD_DATABASE[food]["carbs"] for food in foods]
    fats = [FOOD_DATABASE[food]["fats"] for food in foods]
    
    foods.insert(0, food_name)
    proteins.insert(0, user_macros["protein"])
    carbs.insert(0, user_macros["carbs"])
    fats.insert(0, user_macros["fats"])
    
    df = pd.DataFrame({
        "Food": foods,
        "Protein": proteins,
        "Carbs": carbs,
        "Fats": fats
    })
    
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))
    
    plt.plot(df['Food'], df['Protein'], marker='o', markersize=8, label='Protein', 
             color='#4ECDC4', linewidth=3, linestyle='-', alpha=0.8)
    plt.plot(df['Food'], df['Carbs'], marker='s', markersize=8, label='Carbs', 
             color='#45B7D1', linewidth=3, linestyle='--', alpha=0.8)
    plt.plot(df['Food'], df['Fats'], marker='^', markersize=8, label='Fats', 
             color='#FFC154', linewidth=3, linestyle='-.', alpha=0.8)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.title(f"Macronutrient Trend: {food_name} vs Common Foods", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Food Items", fontsize=12)
    plt.ylabel("Amount (g)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_pie_chart(data):
    fig, ax = plt.subplots(figsize=(2,2))
    plt.style.use('default')
    sns.set_style("white")
    
    labels = ['Protein', 'Carbs', 'Fats']
    values = [data.get("protein", 0), data.get("carbs", 0), data.get("fats", 0)]
    values = [v if not np.isnan(v) else 0 for v in values]
    
    if sum(values) == 0:
        st.warning("⚠️ No valid macronutrient data to plot pie chart.")
        return
    
    explode = [0.05 if v == max(values) else 0 for v in values]
    colors = ['#FF6347', '#3CB371', '#FFD700']
    
    wedges, texts, autotexts = ax.pie(values, explode=explode, labels=labels, 
                                     autopct='%1.1f%%', startangle=140, colors=colors,
                                     textprops={'fontsize': 6}, pctdistance=0.85,
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')  
    plt.title('Macronutrient Distribution', fontsize=10, fontweight='bold', pad=20)
    st.pyplot(fig)
    plt.clf()

def plot_calorie_comparison(user_calories, food_name="Your Food"):
    foods = list(FOOD_DATABASE.keys())
    calories = [FOOD_DATABASE[food]["calories"] for food in foods]
    
    foods.insert(0, food_name)
    calories.insert(0, user_calories)
    
    colors = ["#FF6B6B"] + ["#45B7D1"]*len(FOOD_DATABASE)
    
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    bars = plt.barh(foods, calories, color=colors, edgecolor='white', linewidth=0.7, alpha=0.9)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f"{int(width)} kcal", ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    plt.title(f"Calorie Comparison: {food_name} vs Common Foods", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Calories (kcal)", fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

# ------------------- AI FUNCTIONS ------------------- #
def suggest_healthier_option_gemini(macros, model):
    prompt = f"""
You are a professional nutritionist. A user uploaded food with this nutritional profile:
- Calories: {macros['calories']} kcal
- Protein: {macros['protein']} g
- Carbs: {macros['carbs']} g
- Fats: {macros['fats']} g

Suggest 3 **healthier Food alternatives**. The alternatives must:
- Have **lower calories and fats**
- Include macro values in this format: Calories (kcal), Protein (g), Carbs (g), Fats (g)
- Include a brief explanation of why each alternative is healthier

Respond in bullet points with food names, their macros, and explanations.
"""
    return get_gemini_response(model, prompt).text

def get_food_details(model, food_name, macros, vitamins):
    prompt = f"""
You are a professional nutritionist analyzing {food_name}. Provide detailed information about this food including:

1. **Cultural Origins**: Where does this dish originate from? What cultures traditionally eat it?
2. **Typical Ingredients**: List the main ingredients typically found in this dish
3. **Health Benefits**: Based on its nutritional profile (Calories: {macros['calories']} kcal, Protein: {macros['protein']}g, Carbs: {macros['carbs']}g, Fats: {macros['fats']}g, Vitamins/Minerals: {vitamins}), what are the key health benefits?
4. **Potential Concerns**: Are there any potential health concerns with consuming this food regularly?
5. **Diet Compatibility**: Is this food suitable for: Vegetarian, Vegan, Keto, Gluten-free, Dairy-free diets?

Format your response with clear headings for each section.
"""
    return get_gemini_response(model, prompt).text

def get_recipe_suggestions(model, food_name):
    prompt = f"""
You are a professional chef specializing in healthy cooking. Provide:

1. **Traditional Recipe**: A classic recipe for {food_name} with ingredients and step-by-step instructions
2. **Healthier Variation**: A modified, healthier version of {food_name} with reduced calories/fats
3. **Dietary Adaptations**: How to adapt this recipe for: Vegetarian, Vegan, Keto, Gluten-free diets

Format your response with clear headings and bullet points for ingredients and numbered steps for instructions.
Include approximate preparation and cooking times.
"""
    return get_gemini_response(model, prompt).text

def generate_report(name, macros, gemini_summary, model):
    calorie_pct = macros['calories'] / DAILY_CALORIES * 100
    protein_pct = macros['protein'] / DAILY_MACROS['protein'] * 100
    carbs_pct = macros['carbs'] / DAILY_MACROS['carbs'] * 100
    fats_pct = macros['fats'] / DAILY_MACROS['fats'] * 100

    suggestions = []
    if fats_pct > 70:
        suggestions.append("🔁 Try reducing the oil or butter used during cooking.")
    if carbs_pct > 80:
        suggestions.append("🥗 Consider pairing with a low-carb side like salad or sautéed greens.")
    if calorie_pct > 60:
        suggestions.append("🔥 Opt for grilling or steaming instead of frying.")
    if protein_pct < 40:
        suggestions.append("💪 Add a boiled egg, lentils, or a protein shake to boost protein intake.")

    suggestions_text = "\n".join(suggestions) if suggestions else "✅ This meal looks balanced for your goals!"

    alt_text = "\n### 🍽 Healthier Alternative Suggestions:\n"
    alt_text += suggest_healthier_option_gemini(macros, model)

    return f"""
# Nutrition Report — {name}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

## Nutritional Breakdown
- Calories: {macros['calories']} kcal ({calorie_pct:.1f}% of daily need)
- Protein: {macros['protein']} g ({protein_pct:.1f}%)
- Carbs: {macros['carbs']} g ({carbs_pct:.1f}%)
- Fats: {macros['fats']} g ({fats_pct:.1f}%)

## Healthier Suggestions
{suggestions_text}

{alt_text}

"""

# ------------------- STYLING ------------------- #
st.markdown("""
<style>
    .header-style {
        font-size: 20px;
        font-weight: bold;
        color: #2E86AB;
        padding: 10px;
        border-bottom: 2px solid #2E86AB;
    }
    .metric-card {
        background-color: var(--background-color);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border: 1px solid var(--border-color);
    }
    .food-card {
        border-left: 5px solid #2E86AB;
        padding: 15px;
        background-color: var(--background-color);
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid var(--border-color);
    }
    .detail-card {
        background-color: var(--background-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }
    .stProgress > div > div > div {
        background-color: #2E86AB;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        color: white;
    }
    .tab-content {
        padding: 20px;
        background-color: var(--background-color);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color);
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
    }
    
    body, .stMarkdown, .stText, .stAlert, .stSuccess, .stWarning, .stError {
        color: var(--text-color) !important;
    }
    
    :root {
        --background-color: #F8F9FA;
        --text-color: #000000;
        --border-color: #e0e0e0;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0E1117;
            --text-color: #FFFFFF;
            --border-color: #2d3741;
        }
        
        .metric-card, .food-card, .tab-content, .detail-card {
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
    }
</style>
""", unsafe_allow_html=True)

# ------------------- STREAMLIT APP ------------------- #
# Sidebar for user input
with st.sidebar:
    st.markdown("<p style='text-align: center; color: #2E86AB;font-size: 35px'><b>🍽 Food Analyzer</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader("📸 Upload a food image", type=["jpg", "jpeg", "png"])
    mode = st.radio("Choose input type", ["By Weight (g)", "By Servings"], index=0)
    weight = quantity = None

    if mode == "By Weight (g)":
        weight = st.number_input("Enter weight of food in grams (g):", min_value=1, max_value=10000, value=100)
    else:
        quantity = st.number_input("Enter number of servings:", min_value=1, max_value=100, value=1)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("This AI-powered tool analyzes your food photos and provides detailed nutritional information.")
    st.markdown("---")

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='color: #2E86AB;'>AI-Powered Food Nutrition Dashboard</h1>", unsafe_allow_html=True)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046857.png", width=100)

if uploaded_file:
    start_time = time.time()
    
    with st.spinner("🧠 Analyzing your food image..."):
        # Optimized image processing
        image, temp_path = process_uploaded_image(uploaded_file)
        
        # Display image immediately
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
        
        with col2:
            portion = f"{weight} grams" if weight else f"{quantity} serving(s)"
            
            # Parallel processing
            nutrition_response, _ = analyze_food_parallel(gemini_model, image, portion)
            
            # Process response with cached function
            macros, food_name, vitamins = process_macros(nutrition_response.text)
            
            st.success(f"✅ Analysis complete! (Took {time.time()-start_time:.1f}s)")
            
            # Store current food data
            st.session_state.current_food_data = {
                "name": food_name,
                "macros": macros,
                "nutrition_text": nutrition_response.text,
                "image_path": temp_path,
                "vitamins": vitamins
            }

    if macros is None:
        st.warning("⚠️ No valid nutrition data detected. Please check the image quality or the food item.")
    else:
        # Nutritional Metrics in Cards
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E86AB;'>📊 Nutritional Breakdown: {food_name}</h2>", unsafe_allow_html=True)
        
        # Main calorie card
        with st.container():
            cal_pct = (macros['calories'] / DAILY_CALORIES) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔥 Total Calories</h3>
                <h1>{macros['calories']} kcal</h1>
                <p>{cal_pct:.1f}% of daily intake ({DAILY_CALORIES} kcal)</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(cal_pct/100, 1.0))
        
        # Macronutrient cards in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            protein_pct = (macros['protein'] / DAILY_MACROS['protein']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>🥩 Protein</h3>
                <h2>{macros['protein']} g</h2>
                <p>{protein_pct:.1f}% of daily need</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(protein_pct/100, 1.0))
        
        with col2:
            carbs_pct = (macros['carbs'] / DAILY_MACROS['carbs']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>🍞 Carbohydrates</h3>
                <h2>{macros['carbs']} g</h2>
                <p>{carbs_pct:.1f}% of daily need</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(carbs_pct/100, 1.0))
        
        with col3:
            fats_pct = (macros['fats'] / DAILY_MACROS['fats']) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>🧈 Fats</h3>
                <h2>{macros['fats']} g</h2>
                <p>{fats_pct:.1f}% of daily need</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(fats_pct/100, 1.0))
        
        # Vitamins card
        with st.container():
            st.markdown(f"""
            <div class="food-card">
                <h3>✨ Notable Vitamins/Minerals</h3>
                <p>{st.session_state.current_food_data['vitamins']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E86AB;'>📈 Nutritional Comparisons</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Macronutrient Bar Chart", "Macronutrient Trend", "Calorie Comparison", "Macro Distribution"])
        
        with tab1:
            if tab1:  # Lazy loading
                plot_macro_comparison(macros, food_name)
                
        with tab2:
            if tab2:
                plot_line_comparison(macros, food_name)
                
        with tab3:
            if tab3:
                plot_calorie_comparison(macros['calories'], food_name)
        
        with tab4:
            if tab4:
                plot_pie_chart(macros)

        # Healthier Alternatives
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E86AB;'>🥗 Healthier Alternatives</h2>", unsafe_allow_html=True)
        
        with st.spinner("Finding healthier options..."):
            gemini_alternatives_text = suggest_healthier_option_gemini(macros, gemini_model)
            if "Calories" in gemini_alternatives_text:
                st.markdown(f"""
                <div class="food-card">
                    {gemini_alternatives_text.replace('•', '🍎')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Couldn't find better alternatives for this food.")

        # Food Details Section
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E86AB;'>🍲 Detailed Food Analysis: {food_name}</h2>", unsafe_allow_html=True)
        
        with st.spinner("Generating detailed food analysis..."):
            food_details = get_food_details(gemini_model, food_name, macros, vitamins)
            st.markdown(f"""
            <div class="detail-card">
                {food_details}
            </div>
            """, unsafe_allow_html=True)

        # Recipe Suggestions Section
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E86AB;'>👨‍🍳 Recipe Suggestions for {food_name}</h2>", unsafe_allow_html=True)
        
        with st.spinner("Generating recipe ideas..."):
            recipes = get_recipe_suggestions(gemini_model, food_name)
            st.markdown(f"""
            <div class="detail-card">
                {recipes}
            </div>
            """, unsafe_allow_html=True)

        # Download Report
        st.markdown("---")
        report_text = generate_report(food_name, macros, nutrition_response.text, gemini_model)
        st.download_button("📄 Download Full Nutrition Report", report_text, file_name=f"nutrition_report_{food_name}.txt")

else:
    # Initial empty state
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2 style='color: #2E86AB;'>Welcome to Food Nutrition Analyzer!</h2>
        <p>Upload a food image to get detailed nutritional analysis and AI-powered insights.</p>
        <img src='https://cdn-icons-png.flaticon.com/512/3058/3058971.png' width='200'>
        <p><br>Features include:</p>
        <ul style='text-align: left; display: inline-block;'>
            <li>Nutritional analysis with visuals</li>
            <li>Healthier alternative suggestions</li>
            <li>Detailed food background information</li>
            <li>Recipe suggestions and adaptations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)