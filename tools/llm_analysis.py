import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load miljøvariabler fra .env filen
load_dotenv()

def analyze_with_llm(test_df: pd.DataFrame, target_col: str) -> str:
    """
    Sammenfatter datasættet per dag og beder Groq (Llama-3) om at finde 
    mønstre og perioder, som ser mistænkelige ud.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Fejl: Ingen GROQ_API_KEY fundet i din .env fil. Sørg for at gemme den derinde på en ny linje: GROQ_API_KEY=gsk_..."
        
    client = Groq(api_key=api_key)
    
    # Vi kan ikke sende 8.760 timer til AI'en, så vi regner det daglige forbrug ud
    # for at give en lynhurtig og effektiv oversigt over årets udsving
    daily_summary = test_df[target_col].resample('D').agg(['mean', 'max', 'min']).dropna()
    
    # Oversæt rækkerne til flad tekst som AI'en kan læse
    data_text = ""
    for date, row in daily_summary.iterrows():
        data_text += f"{date.strftime('%Y-%m-%d')}: Snit {row['mean']:.1f}, Max {row['max']:.1f}, Min {row['min']:.1f}\n"
        
    system_prompt = (
        "Du er en teknisk Data Scientist og ekspert i bygningsdrift "
        "(særligt shoppingcentre og varme/strømforbrug). "
        "Du modtager en fil med daglige opsummeringer af et års forbrug. "
        "Din opgave er eksplicit at gennemsøge dataen for ting som matematiske "
        "Maskinlærings-modeller ofte misser: f.eks. "
        "1. Uger/perioder hvor forbruget falder fladt til nul eller næsten 0. "
        "2. Dage med et ekstremt og uforklarligt peak (Max) i forhold til gennemsnittet. "
        "Fremhæv konkret de 3 mest kritiske perioder med DATOER, og forklar HVORFOR "
        "de ser fuldstændigt forkerte ud fra et forretnings- og driftsperspektiv. "
        "Svar kort, professionelt og struktureret på DANSK med pæne markdown bulletpoints."
    )
    
    user_prompt = f"Her er et helt års opsummering for '{target_col}':\n\n{data_text}\n\nHvilke datoer lyser rødt hos dig og hvorfor?"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",  # Opdateret model (70b-8192 er udgået)
            temperature=0.2, # Lidt kreativitet men primært analytisk
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Kunne ikke gennemføre LLM analysen på grund af følgende Cloud-fejl: {str(e)}"

def generate_hourly_alarms(anomalies_df: pd.DataFrame, target_col: str) -> str:
    """
    Tager de specifikke timer hvor Isolation Forest fandt fejl, og beder 
    Llama om at formulere det som en alarm-log (max 15 mest kritiske events).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or anomalies_df.empty:
        return "Ingen data eller API nøgle til rådighed for alarmer."
        
    client = Groq(api_key=api_key)
    
    # Sortér så vi tager de nyeste eller mest ekstreme (hvis vi havde scores, men her tager vi bare de første 20)
    subset = anomalies_df.head(20)
    
    alarm_data = ""
    for idx, row in subset.iterrows():
        alarm_data += f"- Tid: {idx.strftime('%Y-%m-%d %H:%M')}, Værdi: {row[target_col]:.2f}\n"

    system_prompt = (
        "Du er et automatiseret alarm-system for en bygning. "
        "Du modtager en liste over specifikke timer hvor en algoritme har fundet fejl. "
        "Din opgave er at lave en kort, kontant 'Alarm Rapport' på DANSK. "
        "Gruppér timer der ligger tæt op ad hinanden (samme dag/periode). "
        "Forklar kort hvad den driftsansvarlige skal tjekke (f.eks 'Tjek om hovedmåleren er faldet ud' eller 'Højt uforklarligt forbrug om natten'). "
        "Vær meget præcis med klokkeslættene."
    )
    
    user_prompt = f"HER ER ALARM-LISTEN FOR '{target_col}':\n\n{alarm_data}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Alarm-systemet fejlede: {e}"
