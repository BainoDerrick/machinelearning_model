import streamlit as st
import torch
import numpy as np
import pandas as pd
from utils import TeacherGNN, StudentGNN, load_models
import networkx as nx
import os
import sklearn
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ECF Risk Predictor for Ugandan Farmers", page_icon="🐄", layout="wide")

@st.cache_resource
def load_system():
    try:
        teacher, student, kmeans, scaler = load_models()
        logger.info("Models loaded successfully")
        logger.info(f"KMeans cluster_centers_ dtype: {kmeans.cluster_centers_.dtype}")
        logger.info(f"Scaler mean: {scaler['mean']}, std: {scaler['std']}")
        return teacher, student, kmeans, scaler
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

teacher_model, student_model, clusterer, scaler = load_system()

# Risk levels and recommendations
risk_levels = {0: {"name": "Low Risk", "color": "green", "icon": "✅"}, 1: {"name": "High Risk", "color": "red", "icon": "🚨"}}
recommendations = {
    0: [
        "Keep checking your animals for ticks every week.",
        "Use tick spray (acaricide) every 2 weeks to be safe, especially in the rainy season.",
        "Make sure your cattle have clean water and good grass.",
        "Talk to your local vet in your district (e.g., Mbarara, Gulu) if you see any sick animals."
    ],
    1: [
        "EMERGENCY: Spray all your cattle with tick medicine (acaricide) today!",
        "Call your local veterinary officer immediately for help.",
        "Separate any sick animals from the healthy ones to stop the disease from spreading.",
        "Ask your vet about ECF vaccination for your cattle (e.g., through the Uganda Veterinary Association).",
        "Do not let your cattle graze near wetlands or forests where buffaloes live."
    ]
}

# Welcome message
st.title("🐄 East Coast Fever Risk Predictor")
st.markdown("**Welcome, Ugandan Farmer!** This tool helps you keep your cattle safe from East Coast Fever (ECF), a dangerous disease spread by ticks. Answer a few simple questions about your farm, and we’ll tell you if your cattle are at risk.")

# Language selection
language = st.selectbox("Choose Language / Okulonda Olulimi", ["English", "Luganda"])
if language == "Luganda":
    st.markdown("**Tukusanyukidde, Omulimi!** Ekiwandiko kino kikuyamba okukuumira ente zo okuva ku Bulwade bwa East Coast (ECF). Balirira ebibuuzo ebyangu ku mutala gw’ eng'ombe zo, tujja kukutegeeza oba ente zo ziri mu kabi.")

# Show sample table format for file upload
st.markdown("### File Format for Uploading Data" if language == "English" else "### Ekyokulabirako Eky’ekiwandiiko Eky’okutikka")
st.markdown("""
| Ticks | Buffaloes | Cattle | Temperature |
|-------|-----------|--------|-------------|
| 0.5   | 0.005     | 45.0   | 29.0        |
""")
st.markdown(
    "If you’re uploading a file, make sure it matches the format above with columns: Ticks, Buffaloes, Cattle, Temperature. The system will not work if the format is different."
    if language == "English"
    else "Obanga otikka fayiro, gulumiza nti efanana n’ekyokulabirako waggulu nga erina ebibagiro: Enkwa, Embogo, Ente, Obutiti. Ekikozesebwa tekizakola obanga ekika ky’ekiwandiiko ky’enjawulo."
)

# Input section
with st.expander("📝 Tell Us About Your Farm" if language == "English" else "📝 Tugambe Ku Mutala Gw’ Ente Zo", expanded=True):
    tab1, tab2 = st.tabs(["Enter Details" if language == "English" else "Wandiika Ebikwata ku Mutala", "Upload File" if language == "English" else "Tikka Fayiro"])
    
    with tab1:
        st.markdown("### What Do You See on Your Farm?" if language == "English" else "### Okirabye Ki Ku Mutala Gw’ Ente Zo?")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tick_label = "Average Ticks on Your Animals 🕷️" if language == "English" else "Obungi bw’ Enkwa Ku Nte zo 🕷️"
            tick_help = "Count ticks on 5 animals, add them up, and divide by 5. Example: 2 + 1 + 3 + 0 + 1 = 7, so 7 ÷ 5 = 1.4. If the average is more than 1.0, enter 1.0 (the maximum). Enter 0 if no ticks." if language == "English" else "Balirira enkwa ku nte 5, ziwerekere, ogabanye mu 5. Okyokulabirako: 2 + 1 + 3 + 0 + 1 = 7, kale 7 ÷ 5 = 1.4. Obanga obungi bwa enkwa busukka 1.0, wandiika 1.0 (ekisukka). Wandiika 0 obanga tewali nkwa."
            tick = st.number_input(
                tick_label,
                min_value=0.0,
                max_value=1.0,
                value=0.015,
                step=0.001,
                format="%.3f",
                help=tick_help
            )
        with col2:
            cape_label = "Buffaloes Seen Near Your Farm 🐃" if language == "English" else "Embogo Ezirabise Okumpi N’omutala Gw’ Ente Zo 🐃"
            cape_help = "How many buffaloes have you seen near your farm in the last 30 days? If you’ve seen 1 or more, enter 0.01 (the maximum). Enter 0 if none." if language == "English" else "Embogo z’ omwezi guno z’ olabye okumpi n’ omutala gw’ ente zo ziri mmeka? Obanga olabye 1 oba eziwera, wandiika 0.01 (ekisukka). Wandiika 0 obanga tewali."
            cape = st.number_input(
                cape_label,
                min_value=0.0,
                max_value=0.01,
                value=0.0004,
                step=0.0001,
                format="%.4f",
                help=cape_help
            )
        with col3:
            cattle_label = "Total Cattle in Your Grazing Area 🐄" if language == "English" else "Ente Zonna Eziri Mu Mutala Gw’ Okulundirira 🐄"
            cattle_help = "Count all cattle (yours and neighbors’) that graze together. Example: You have 10, your neighbor has 5 = 15 total." if language == "English" else "Balirira ente zonna (ezizo n’ eza baliraanwa bo) ezilundirira wamu. Okyokulabirako: Olina 10, omuliraanwa wo alina 5 = 15 zonna."
            cattle = st.number_input(
                cattle_label,
                min_value=0.0,
                max_value=150.0,
                value=16.0,
                step=0.1,
                format="%.1f",
                help=cattle_help
            )
        with col4:
            bio5_label = "Daytime Temperature (°C) 🌡️" if language == "English" else "Obutiti bw’ Ekifudde Mu Nnaku (Digiri Senteegureedi) 🌡️"
            bio5_help = "Enter the daytime temperature. Guess if you don’t know: Very hot = 30, Warm = 28, Cool = 25." if language == "English" else "Wandiika obutiti bw’ ekifudde mu nnaku. Bujjira obanga tolimanya: Ekifudde kya maanyi = 30, Ekifudde kya bulijjo = 28, Ekifudde ekirimu = 25."
            bio5 = st.number_input(
                bio5_label,
                min_value=24.0,
                max_value=35.0,
                value=30.49,
                step=0.01,
                format="%.2f",
                help=bio5_help
            )
        submitted = st.button("Check Risk Level" if language == "English" else "Kebera Obuzibu")

    with tab2:
        upload_label = "Upload your farm data (CSV/Excel)" if language == "English" else "Tikka fayiro ey’ ekwata ku mutala gw’ eng'ombe zo (CSV/Excel)"
        upload_help = "File should have columns: Ticks, Buffaloes, Cattle, Temperature. You can include multiple rows for different days or observations." if language == "English" else "Fayiro erina okuba n’ ebibagiro: Enkwa, Embogo, Ente, Obutiti. Oyinza okuwandiika ebibagiro bingi nga by’ ennaku ez’enjawulo."
        uploaded_file = st.file_uploader(upload_label, type=["csv", "xlsx"], help=upload_help)
        if uploaded_file:
            try:
                # Read the file (CSV or Excel)
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                
                # Validate that required columns exist
                required_columns = ['ticks', 'buffaloes', 'cattle', 'temperature']
                found_columns = {col.lower(): col for col in df.columns}
                missing_columns = [col for col in required_columns if not any(col in found_col for found_col in found_columns.keys())]
                if missing_columns:
                    raise ValueError(f"File format is incorrect. Missing columns: {', '.join(missing_columns)}. Please use the format shown above with columns: Ticks, Buffaloes, Cattle, Temperature.")
                
                # Map column names to standardized names
                data = []
                for _, row in df.iterrows():
                    row_data = {}
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'tick' in col_lower:
                            row_data['ticks'] = float(row[col])
                        elif 'buffalo' in col_lower:
                            row_data['buffaloes'] = float(row[col])
                        elif 'cattle' in col_lower:
                            row_data['cattle'] = float(row[col])
                        elif 'temp' in col_lower:
                            row_data['temperature'] = float(row[col])
                    data.append(row_data)
                
                # Convert to DataFrame for processing
                input_df = pd.DataFrame(data)
                
                # Store predictions for each row
                results = []
                for idx, row in input_df.iterrows():
                    tick = row['ticks']
                    cape = row['buffaloes']
                    cattle = row['cattle']
                    bio5 = row['temperature']
                    
                    # Apply the same prediction logic as manual input
                    features_raw = np.array([[tick, cape, cattle, bio5]], dtype=np.float64)
                    logger.info(f"Row {idx} - Raw features: {features_raw}")
                    
                    features_scaled = (features_raw - scaler['mean']) / scaler['std']
                    logger.info(f"Row {idx} - Scaled features: {features_scaled}")
                    
                    features_kmeans = features_scaled.astype(np.float32)
                    cluster = clusterer.predict(features_kmeans)[0]
                    logger.info(f"Row {idx} - Cluster: {cluster}")
                    
                    cluster_array = np.array([[cluster]], dtype=np.float32)
                    full_features = np.concatenate([features_scaled.astype(np.float32), cluster_array], axis=1)
                    logger.info(f"Row {idx} - Full features: {full_features}")
                    
                    input_tensor = torch.tensor(full_features, dtype=torch.float32)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    
                    with torch.no_grad():
                        student_out = student_model(input_tensor, edge_index)
                        probs = student_out.softmax(dim=1).numpy()[0]
                        final_pred = 1 if probs[1] > 0.5 else 0
                        risk_info = risk_levels[final_pred]
                    logger.info(f"Row {idx} - Student output (logits): {student_out.numpy()}")
                    logger.info(f"Row {idx} - Probabilities: ECF=0: {probs[0]:.4f}, ECF=1: {probs[1]:.4f}")
                    logger.info(f"Row {idx} - Predicted class: {final_pred}")
                    
                    # Store the result
                    results.append({
                        'Ticks': tick,
                        'Buffaloes': cape,
                        'Cattle': cattle,
                        'Temperature': bio5,
                        'Risk Level': risk_info['name'],
                        'Confidence (%)': f"{float(probs[final_pred]) * 100:.1f}",
                        'Recommendations': "; ".join(recommendations[final_pred])
                    })
                
                # Display results in a table
                results_df = pd.DataFrame(results)
                st.success("File loaded and processed successfully!" if language == "English" else "Fayiro etikkiddwa era ekolwa bulungi!")
                st.markdown("### Results for Each Observation" if language == "English" else "### Ebyava mu Kulondoola Kwo")
                st.dataframe(results_df)
                
                # Allow the farmer to save the results
                save_btn = "📥 Save All Results" if language == "English" else "📥 Kaza Ebyava Byonna"
                save_msg = "Results saved successfully!" if language == "English" else "Ebyava byakaziddwa bulungi!"
                if st.button(save_btn):
                    results_df.to_csv(f"ecf_results_{pd.Timestamp.now().date()}.csv", index=False)
                    st.success(save_msg)
                
                submitted = False  # Prevent the manual prediction logic from running
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}" if language == "English" else f"Ensobi mu kutikka fayiro: {str(e)}")

if submitted:
    try:
        features_raw = np.array([[tick, cape, cattle, bio5]], dtype=np.float64)
        logger.info(f"Raw features: {features_raw}")
        
        features_scaled = (features_raw - scaler['mean']) / scaler['std']
        logger.info(f"Scaled features: {features_scaled}")
        
        features_kmeans = features_scaled.astype(np.float32)
        cluster = clusterer.predict(features_kmeans)[0]
        logger.info(f"Cluster: {cluster}")
        
        cluster_array = np.array([[cluster]], dtype=np.float32)
        full_features = np.concatenate([features_scaled.astype(np.float32), cluster_array], axis=1)
        logger.info(f"Full features: {full_features}")
        
        input_tensor = torch.tensor(full_features, dtype=torch.float32)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        with torch.no_grad():
            student_out = student_model(input_tensor, edge_index)
            probs = student_out.softmax(dim=1).numpy()[0]
            final_pred = 1 if probs[1] > 0.5 else 0
            risk_info = risk_levels[final_pred]
        logger.info(f"Student output (logits): {student_out.numpy()}")
        logger.info(f"Probabilities: ECF=0: {probs[0]:.4f}, ECF=1: {probs[1]:.4f}")
        logger.info(f"Predicted class: {final_pred}")
        
        st.markdown("---")
        pred_text = f"## {risk_info['icon']} Prediction: **<span style='color:{risk_info['color']}'>{risk_info['name']}</span>**" if language == "English" else f"## {risk_info['icon']} Okutegeera: **<span style='color:{risk_info['color']}'>{risk_info['name']}</span>**"
        st.markdown(pred_text, unsafe_allow_html=True)
        
        with st.expander("🔍 How Sure Are We?" if language == "English" else "🔍 Tuli W’ Obukakafu Bw’ amodel?"):
            student_conf = float(probs[final_pred])
            conf_label = "System Confidence" if language == "English" else "Obukakafu bw’ Ekikozesebwa"
            st.metric(conf_label, f"{student_conf*100:.1f}%")
            st.progress(student_conf)
            
            # Explain the prediction in simple terms
            explanation = "We think this because:\n"
            if language == "Luganda":
                explanation = "Tukirowooza kino kubanga:\n"
            
            if final_pred == 1:  # High Risk
                reasons = []
                if tick > 0.167:
                    reasons.append("- You have many ticks on your animals." if language == "English" else "- Olina enkwa nnyingi ku nte zo.")
                if cape > 0.00088:
                    reasons.append("- There are buffaloes near your farm." if language == "English" else "- Waliwo embogo okumpi n’ omutala gw’ ente zo.")
                if cattle > 42.19:
                    reasons.append("- There are many cattle in your grazing area." if language == "English" else "- Waliwo ente nnyingi mu mutala gw’ okulundirira.")
                if bio5 > 29.82:
                    reasons.append("- It is very hot, which helps ticks grow." if language == "English" else "- Ebugumu lya maanyi, ekyongera enkwa.")
                explanation += "\n".join(reasons) if reasons else "We see some risks on your farm." if language == "English" else "Tulaba ebizibu ku mutala gw’ ente zo."
            else:  # Low Risk
                explanation += "- Your farm looks safe right now. You have few ticks, few buffaloes nearby, and the weather is not too hot." if language == "English" else "- Omutala gw’ ente zo gulabika nga guli bulungi kati. Olina enkwa ntono, embogo ntono okumpi, n’ embera yo'budde si ya bugumu nnyo."
            st.markdown(explanation)
        
        rec_title = "## 🛡️ What You Should Do" if language == "English" else "## 🛡️ Kiki Ekirina Okukolebwa"
        st.markdown(rec_title)
        for i, action in enumerate(recommendations[final_pred]):
            st.markdown(f"{i+1}. {action}")
        
        save_btn = "📥 Save This Report" if language == "English" else "📥 Savinga Alipoota Eno"
        save_msg = "Report saved successfully!" if language == "English" else "Alipoota Esavingidwa bulungi!"
        if st.button(save_btn):
            report = {"Date": pd.Timestamp.now(), "Risk Level": risk_info['name'], "Ticks": tick, "Buffaloes": cape, "Cattle": cattle, "Temperature": bio5, "Prediction": final_pred}
            pd.DataFrame([report]).to_csv(f"ecf_report_{pd.Timestamp.now().date()}.csv", index=False)
            st.success(save_msg)
        
    except Exception as e:
        err_msg = f"Prediction failed: {str(e)}" if language == "English" else f"Okutegeera kwavuwalidde: {str(e)}"
        st.error(err_msg)
        logger.error(f"Prediction error: {str(e)}")

# Farming Tips & Education
with st.expander("📚 Farming Tips & Education" if language == "English" else "📚 Ebyokuyiga Ebikwata ku Kulima"):
    st.markdown("### Watch These Videos to Learn More" if language == "English" else "### Kebera Vidiyo Zino Okuyiga Ebisingawo")
    
    # Create three columns for the videos
    col1, col2, col3 = st.columns(3)
    
    # Video 1: ECF Overview
    with col1:
        st.video("https://youtu.be/YUBIwKaW3Oc?si=IaCdMMwrmEL7FfdI", start_time=0)
        st.caption("ECF Overview" if language == "English" else "Okulabula ku ECF")
    
    # Video 2: Tick Control in Cattle
    with col2:
        st.video("https://youtu.be/Gr7nHyABmqY?si=yBhNDGD2LucaaHF3", start_time=0)
        st.caption("Tick Control in Cattle" if language == "English" else "Okulwanyisa Enkwa mu Nte")
    
    # Video 3: Cattle Health Tips for Ugandan Farmers
    with col3:
        st.video("https://youtu.be/38yP76pA82c?si=s7uO74CvMOz8fmmc", start_time=0)
        st.caption("Cattle Health Tips for Ugandan Farmers" if language == "English" else "Ebyokulabirako ku Bulamu bw’Ente eri Abalimi b’e Uganda")
    
    st.markdown("""
    ### Tips to Keep Your Cattle Safe
    - **What is ECF?** A deadly cattle disease spread by ticks. It can kill your animals in days if not treated.  
    - **Prevention:**  
      - Spray your cattle with tick medicine (acaricide) every 2 weeks, especially in the rainy season (e.g., March-May in Uganda).  
      - Keep your grazing area clean—remove tall grass where ticks hide, especially in wetland areas.  
      - Avoid grazing near forests or game parks (e.g., near Queen Elizabeth National Park) where buffaloes live.  
    - **Symptoms to Watch For:**  
      - High fever (your cow feels very hot).  
      - Swollen neck or legs.  
      - Your cow stops eating or looks weak.  
    - **Where to Get Help:**  
      - Visit your nearest veterinary office in your district (e.g., Kampala, Gulu, or Mbarara).  
      - Join a local farmer group like the Uganda National Farmers Federation (UNFFE) to learn more about ECF prevention.  
      - Call the Uganda Veterinary Helpline at 0800-123-456 for free advice (placeholder—replace with a real number).
    """ if language == "English" else """
    ### Ebyokuyiga Ebikwata ku Kulima
    - **ECF Kiki?** Bulwadde bwa maanyi mu nte obusasanyizibwa enkwa. Buyinza okutta ente zo mu nnaku ntono wezibanga tezitwaliddwa mu ddwaliro.  
    - **Okuziyiza:**  
      - Fuyila ente zo neddagala ly’ enkwa (acaricide) buli wiiki 2, naddala mu kiseera ky’ enkuba (nga March-May mu Uganda).  
      - Kuumira omutala gw’ okulundira nga guli bulungi—ggyawo obusubi ewabela enkwa, naddala mu bifo ebyentobazi.  
      - Tolisiza nte zo  okumpi n’ ebibira oba ppaaka (nga Queen Elizabeth National Park) embogo wezibeera.  
    - **Ebikwata ku Bwasika:**  
      - Omusujja ogwa maanyi (eng'ombe yo efudde nnyo).  
      - Okuzimba mu bulago oba mu magulu.  
      - Ente yo ekomawo okulya oba erabika nga teyina manyi.  
    - **W’ Oyinza Okufuna Obuyambi:**  
      - Genda mu ofiisi y’ eddagala ly’ emyala ey’ okumpi mu disitulikiti yo (nga Kampala, Gulu, oba Mbarara).  
      - Yingira mu kibiina ky’ abalimi mu bitundu byo nga Uganda National Farmers Federation (UNFFE) okuyiga ebisingawo ku kweziza ECF.  
      - Telefoni ku Uganda Veterinary Helpline ku 0800-123-456 okufuna okubuulirira kwa bwereere (namba ey’ okukozesa—sindika ey’ eddala).
    """)

# Feedback button
st.markdown("---")
feedback_label = "💬 Tell Us If This Helped!" if language == "English" else "💬 Tugambe Obanga Kino Kikuyambye!"
if st.button(feedback_label):
    st.text_area("Your Feedback / Ebyokwogera Byo", placeholder="Did the prediction help you? Any problems?" if language == "English" else "Okutegeera kwakuyamba? Waliwo ebizibu?")
    st.success("Thank you for your feedback!" if language == "English" else "Webale kwebyokwogera byo!")

# Footer
st.markdown("""
**Need help?** Contact your local veterinary officer  
**Emergency:** Call the Uganda Veterinary Helpline at 0800-123-456   
Built with ❤️ for Ugandan farmers
""" if language == "English" else """
**Oyinza okwetaaga obuyambi?** Saba obuyambi ku muweereza w’ eddagala ly’ emyala mu bitundu byo  
**Eby’ okuddaabiriza:** Kubba ku Uganda Veterinary Helpline ku 0800-123-456  
Ekikozesebwa kino kyakolebwa n’ omukwano ❤️ eri abalimi b’ e Uganda
""")