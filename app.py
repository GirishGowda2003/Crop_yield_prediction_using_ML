"""app.py - Crop Yield Prediction Web Application v3.2
FIXED - Exact UI Match + Rainfall + Header Space Fixes + Multi-Language Support
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('dark_background')
sns.set_palette("husl")

# Page config
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Modern CSS
st.markdown("<style>" + open("style.css").read() + "</style>", unsafe_allow_html=True)

# Language translations
TRANSLATIONS = {
    'en': {
        'title': '🌾 CROP YIELD PREDICTOR',
        'subtitle': 'AGRICULTURAL ANALYTICS WITH DISTRICT & RAINFALL INSIGHTS',
        'enter_farm_details': '📝 Enter Farm Details',
        'state': '🗺️ Select State',
        'district': '🏘️ Select District',
        'crop': '🌱 Select Crop',
        'season': '🌤️ Select Season',
        'area': '📐 Farm Area (Acres)',
        'area_help': 'Enter total farm area in acres',
        'year': '📅 Crop Year',
        'year_help': 'Select year for prediction',
        'rainfall': '🌧️ Annual Rainfall (mm)',
        'rainfall_help': 'Average rainfall for {district}: {rainfall:.1f} mm',
        'predict_btn': '🚀 PREDICT YIELD NOW',
        'historical_insights': '📊 Historical Insights',
        'avg_yield': '📊 Avg Yield',
        'max_yield': '🏆 Max Yield',
        'records': '📝 Records',
        'avg_rainfall': '🌧️ Avg Rainfall',
        'district_label': '📍 District',
        'no_data': 'ℹ️ No historical data for {crop} in {district}',
        'no_load': '⚠️ Could not load historical data',
        'state_district': '{state} • {district} • {crop} • {season} • {year}',
        'area_card': '📐 Area',
        'yield_acre': '🎯 Yield/Acre',
        'yield_ha': '🎯 Yield/Ha',
        'production': '📦 Production',
        'rainfall_card': '🌧️ Rainfall',
        'production_summary': '📊 Production Summary',
        'area_metric': 'Area',
        'hectares': 'Hectares',
        'yield_acre_metric': 'Yield/Acre',
        'production_metric': 'Production',
        'district_metric': 'District',
        'detailed_analysis': '📈 Detailed Analysis & Insights',
        'metrics_overview': 'Production Metrics Overview',
        'comparison': 'Comparison - {district}',
        'higher': '✅ {pct:.1f}% higher than district average',
        'lower': '⚠️ {pct:.1f}% lower than district average',
        'equal': 'ℹ️ Equal to district average',
        'comp_unavailable': '📊 Comparison unavailable',
        'district_comparison': '🏘️ District-wise Comparison in {state}',
        'top_districts': 'Top 10 Districts for {crop}',
        'rainfall_vs_yield': 'Rainfall vs Yield - {state}',
        'district_comp_unavailable': '📊 District comparison unavailable',
        'rainfall_analysis_unavailable': '📊 Rainfall analysis unavailable',
        'detailed_calc': '💡 View Detailed Calculation Breakdown',
        'input_params': '📐 Input Parameters',
        'yield_calc': '🎯 Yield Calculations',
        'production_calc': '📦 Production Calculations',
        'conversions': '💡 Conversions',
        'success': '✅ Prediction completed successfully!',
        'error': '❌ Prediction Error: {error}',
        'check_inputs': '💡 Please check your input values',
        'footer_title': '🌾 Crop Yield Predictor v3.2',
        'footer_subtitle': 'With District-wise Analysis & Rainfall Intelligence',
        'footer_copyright': '© 2025 • Powered by Advanced Machine Learning & Data Science',
        'analyzing': '🔮 Analyzing agricultural data...',
        'no_hist_data': "No historical data for {crop} in {district}"
    },
    'hi': {
        'title': '🌾 फसल उपज भविष्यवक्ता',
        'subtitle': 'जिला और वर्षा अंतर्दृष्टि के साथ संचालित कृषि विश्लेषण',
        'enter_farm_details': '📝 खेत विवरण दर्ज करें',
        'state': '🗺️ राज्य चुनें',
        'district': '🏘️ जिला चुनें',
        'crop': '🌱 फसल चुनें',
        'season': '🌤️ मौसम चुनें',
        'area': '📐 खेत क्षेत्र (एकड़)',
        'area_help': 'कुल खेत क्षेत्र एकड़ में दर्ज करें',
        'year': '📅 फसल वर्ष',
        'year_help': 'भविष्यवाणी के लिए वर्ष चुनें',
        'rainfall': '🌧️ वार्षिक वर्षा (मिमी)',
        'rainfall_help': '{district} के लिए औसत वर्षा: {rainfall:.1f} मिमी',
        'predict_btn': '🚀 अभी उपज भविष्यवाणी करें',
        'historical_insights': '📊 ऐतिहासिक जानकारी',
        'avg_yield': '📊 औसत उपज',
        'max_yield': '🏆 अधिकतम उपज',
        'records': '📝 रिकॉर्ड',
        'avg_rainfall': '🌧️ औसत वर्षा',
        'district_label': '📍 जिला',
        'no_data': '{district} में {crop} के लिए कोई ऐतिहासिक डेटा नहीं',
        'no_load': '⚠️ ऐतिहासिक डेटा लोड नहीं हो सका',
        'state_district': '{state} • {district} • {crop} • {season} • {year}',
        'area_card': '📐 क्षेत्र',
        'yield_acre': '🎯 उपज/एकड़',
        'yield_ha': '🎯 उपज/हेक्टेयर',
        'production': '📦 उत्पादन',
        'rainfall_card': '🌧️ वर्षा',
        'production_summary': '📊 उत्पादन सारांश',
        'area_metric': 'क्षेत्र',
        'hectares': 'हेक्टेयर',
        'yield_acre_metric': 'उपज/एकड़',
        'production_metric': 'उत्पादन',
        'district_metric': 'जिला',
        'detailed_analysis': '📈 विस्तृत विश्लेषण और अंतर्दृष्टि',
        'metrics_overview': 'उत्पादन मेट्रिक्स अवलोकन',
        'comparison': '{district} - तुलना',
        'higher': '✅ जिला औसत से {pct:.1f}% अधिक',
        'lower': '⚠️ जिला औसत से {pct:.1f}% कम',
        'equal': 'ℹ️ जिला औसत के बराबर',
        'comp_unavailable': '📊 तुलना उपलब्ध नहीं',
        'district_comparison': '{state} में जिला-वार तुलना',
        'top_districts': '{crop} के लिए शीर्ष 10 जिले',
        'rainfall_vs_yield': '{state} - वर्षा बनाम उपज',
        'district_comp_unavailable': '📊 जिला तुलना उपलब्ध नहीं',
        'rainfall_analysis_unavailable': '📊 वर्षा विश्लेषण उपलब्ध नहीं',
        'detailed_calc': '💡 विस्तृत गणना ब्रेकडाउन देखें',
        'input_params': '📐 इनपुट पैरामीटर',
        'yield_calc': '🎯 उपज गणनाएँ',
        'production_calc': '📦 उत्पादन गणनाएँ',
        'conversions': '💡 रूपांतरण',
        'success': '✅ भविष्यवाणी सफलतापूर्वक पूरी!',
        'error': '❌ भविष्यवाणी त्रुटि: {error}',
        'check_inputs': '💡 कृपया अपने इनपुट मान जांचें',
        'footer_title': '🌾 फसल उपज भविष्यवक्ता v3.2',
        'footer_subtitle': 'जिला-वार विश्लेषण और वर्षा बुद्धिमत्ता के साथ',
        'footer_copyright': '© 2025 • उन्नत मशीन लर्निंग और डेटा साइंस द्वारा संचालित'
    },
    'kn': {
        'title': '🌾 ಬೆಳೆ ಇಳುವರಿ ಮುನುಪಡೆ',
        'subtitle': 'ಜಿಲ್ಲೆ ಮತ್ತು ಮಳೆ ಬುದ್ಧಿಮತ್ತ ಜೊತೆಗೆ ಸಬಲೀಕರಿಸಿದ ಕೃಷಿ ವಿಶ್ಲೇಷಣೆ',
        'enter_farm_details': '📝 ಫಾರ್ಮ್ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ',
        'state': '🗺️ ರಾಜ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'district': '🏘️ ಜಿಲ್ಲೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'crop': '🌱 ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'season': '🌤️ ಋತುವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'area': '📐 ಫಾರ್ಮ್ ಪ್ರದೇಶ (ಎಕರೆಗಳು)',
        'area_help': 'ಒಟ್ಟು ಫಾರ್ಮ್ ಪ್ರದೇಶವನ್ನು ಎಕರೆಗಳಲ್ಲಿ ನಮೂದಿಸಿ',
        'year': '📅 ಬೆಳೆ ವರ್ಷ',
        'year_help': 'ಮುನುಪಡೆಗೆ ವರ್ಷವನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'rainfall': '🌧️ ವಾರ್ಷಿಕ ಮಳೆ (ಮಿ.ಮೆ.)',
        'rainfall_help': '{district} ಗೆ ಸರಾಸರಿ ಮಳೆ: {rainfall:.1f} ಮಿ.ಮೆ.',
        'predict_btn': '🚀 ಈಗ ಇಳುವರಿ ಮುನುಪಡೆ ಮಾಡಿ',
        'historical_insights': '📊 ಐತಿಹಾಸಿಕ ಮಾಹಿತಿ',
        'avg_yield': '📊 ಸರಾಸರಿ ಇಳುವರಿ',
        'max_yield': '🏆 ಗರಿಷ್ಠ ಇಳುವರಿ',
        'records': '📝 ದಾಖಲೆಗಳು',
        'avg_rainfall': '🌧️ ಸರಾಸರಿ ಮಳೆ',
        'district_label': '📍 ಜಿಲ್ಲೆ',
        'no_data': '{district} ನಲ್ಲಿ {crop} ಗೆ ಯಾವುದೇ ಐತಿಹಾಸಿಕ ಡೇಟಾ ಇಲ್ಲ',
        'no_load': '⚠️ ಐತಿಹಾಸಿಕ ಡೇಟಾ ಲೋಡ್ ಮಾಡಲಿಲ್ಲ',
        'state_district': '{state} • {district} • {crop} • {season} • {year}',
        'area_card': '📐 ಪ್ರದೇಶ',
        'yield_acre': '🎯 ಇಳುವರಿ/ಎಕರೆ',
        'yield_ha': '🎯 ಇಳುವರಿ/ಹೆಕ್ಟೇರ್',
        'production': '📦 ಉತ್ಪಾದನೆ',
        'rainfall_card': '🌧️ ಮಳೆ',
        'production_summary': '📊 ಉತ್ಪಾದನೆ ಸಾರಾಂಶ',
        'area_metric': 'ಪ್ರದೇಶ',
        'hectares': 'ಹೆಕ್ಟೇರ್',
        'yield_acre_metric': 'ಇಳುವರಿ/ಎಕರೆ',
        'production_metric': 'ಉತ್ಪಾದನೆ',
        'district_metric': 'ಜಿಲ್ಲೆ',
        'detailed_analysis': '📈 ವಿವರಣಾತ್ಮಕ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಒಳನೋಟಗಳು',
        'metrics_overview': 'ಉತ್ಪಾದನೆ ಮೆಟ್ರಿಕ್ಸ್ ಅವಲೋಕನ',
        'comparison': '{district} - ಹೋಲಿಕೆ',
        'higher': '✅ ಜಿಲ್ಲೆ ಸರಾಸರಿಯಿಂದ {pct:.1f}% ಹೆಚ್ಚು',
        'lower': '⚠️ ಜಿಲ್ಲೆ ಸರಾಸರಿಯಿಂದ {pct:.1f}% ಕಡಿಮೆ',
        'equal': 'ℹ️ ಜಿಲ್ಲೆ ಸರಾಸರಿಗೆ ಸಮಾನ',
        'comp_unavailable': '📊 ಹೋಲಿಕೆ ಲಭ್ಯವಿಲ್ಲ',
        'district_comparison': '{state} ನಲ್ಲಿ ಜಿಲ್ಲೆ-ಆಧಾರಿತ ಹೋಲಿಕೆ',
        'top_districts': '{crop} ಗೆ ಟಾಪ್ 10 ಜಿಲ್ಲೆಗಳು',
        'rainfall_vs_yield': '{state} - ಮಳೆ vs ಇಳುವರಿ',
        'district_comp_unavailable': '📊 ಜಿಲ್ಲೆ ಹೋಲಿಕೆ ಲಭ್ಯವಿಲ್ಲ',
        'rainfall_analysis_unavailable': '📊 ಮಳೆ ವಿಶ್ಲೇಷಣೆ ಲಭ್ಯವಿಲ್ಲ',
        'detailed_calc': '💡 ವಿವರಣಾತ್ಮಕ ಲೆಕ್ಕಾಚಾರ ಬ್ರೇಕ್‌ಡೌನ್ ವೀಕ್ಷಿಸಿ',
        'input_params': '📐 ಇನ್‌ಪುಟ್ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು',
        'yield_calc': '🎯 ಇಳುವರಿ ಲೆಕ್ಕಾಚಾರಗಳು',
        'production_calc': '📦 ಉತ್ಪಾದನೆ ಲೆಕ್ಕಾಚಾರಗಳು',
        'conversions': '💡 ಪರಿವರ್ತನೆಗಳು',
        'success': '✅ ಮುನುಪಡೆ ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಂಡಿದೆ!',
        'error': '❌ ಮುನುಪಡೆ ತಪ್ಪು: {error}',
        'check_inputs': '💡 ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಮೌಲ್ಯಗಳನ್ನು ಪರಿಶೀಲಿಸಿ',
        'footer_title': '🌾 ಬೆಳೆ ಇಳುವರಿ ಮುನುಪಡೆ v3.2',
        'footer_subtitle': 'ಜಿಲ್ಲೆ-ಆಧಾರಿತ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಮಳೆ ಬುದ್ಧಿಮತ್ತೆಯೊಂದಿಗೆ',
        'footer_copyright': '© 2025 • ಅಧುನತ ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಮತ್ತು ಡೇಟಾ ಸೈನ್ಸ್ ಅಧಾರಿತ'
    },
    'ta': {
        'title': '🌾 பயிர் விளைச்சல் கணிப்பான்',
        'subtitle': 'மாவட்டம் & மழை புலப்படுத்தலுடன் ஆன கால்நடை பகுப்பாய்வு',
        'enter_farm_details': '📝 விவசாய விவரங்களை உள்ளிடவும்',
        'state': '🗺️ மாநிலத்தை தேர்ந்தெடுக்கவும்',
        'district': '🏘️ மாவட்டத்தை தேர்ந்தெடுக்கவும்',
        'crop': '🌱 பயிரை தேர்ந்தெடுக்கவும்',
        'season': '🌤️ பருவத்தை தேர்ந்தெடுக்கவும்',
        'area': '📐 விவசாய பரப்பு (ஏக்கர்)',
        'area_help': 'மொத்த விவசாய பரப்பை ஏக்கரில் உள்ளிடவும்',
        'year': '📅 பயிர் ஆண்டு',
        'year_help': 'கணிப்பிற்கான ஆண்டை தேர்ந்தெடுக்கவும்',
        'rainfall': '🌧️ ஆண்டு மழை (மி.மி)',
        'rainfall_help': '{district} சராசரி மழை: {rainfall:.1f} மி.மி',
        'predict_btn': '🚀 இப்போது விளைச்சலை கணிக்கவும்',
        'historical_insights': '📊 வரலாற்று தகவல்கள்',
        'avg_yield': '📊 சராசரி விளைச்சல்',
        'max_yield': '🏆 அதிகபட்ச விளைச்சல்',
        'records': '📝 பதிவுகள்',
        'avg_rainfall': '🌧️ சராசரி மழை',
        'district_label': '📍 மாவட்டம்',
        'no_data': '{district} இல் {crop} க்கு வரலாற்று தரவு இல்லை',
        'no_load': '⚠️ வரலாற்று தரவு ஏற்ற முடியவில்லை',
        'state_district': '{state} • {district} • {crop} • {season} • {year}',
        'area_card': '📐 பரப்பு',
        'yield_acre': '🎯 விளைச்சல்/ஏக்கர்',
        'yield_ha': '🎯 விளைச்சல்/ஹெக்டேர்',
        'production': '📦 உற்பத்தி',
        'rainfall_card': '🌧️ மழை',
        'production_summary': '📊 உற்பத்தி சுருக்கம்',
        'area_metric': 'பரப்பு',
        'hectares': 'ஹெக்டேர்',
        'yield_acre_metric': 'விளைச்சல்/ஏக்கர்',
        'production_metric': 'உற்பத்தி',
        'district_metric': 'மாவட்டம்',
        'detailed_analysis': '📈 விரிவான பகுப்பாய்வு & உள்ளுணர்வுகள்',
        'metrics_overview': 'உற்பத்தி அளவுகள் கண்ணோட்டம்',
        'comparison': '{district} - ஒப்பீடு',
        'higher': '✅ மாவட்ட சராசரியை விட {pct:.1f}% அதிகம்',
        'lower': '⚠️ மாவட்ட சராசரியை விட {pct:.1f}% குறைவு',
        'equal': 'ℹ️ மாவட்ட சராசரியுக்கு சமம்',
        'comp_unavailable': '📊 ஒப்பீடு கிடைக்கவில்லை',
        'district_comparison': '{state} இல் மாவட்ட-வாரியாக ஒப்பீடு',
        'top_districts': '{crop} க்கு முதல் 10 மாவட்டங்கள்',
        'rainfall_vs_yield': '{state} - மழை vs விளைச்சல்',
        'district_comp_unavailable': '📊 மாவட்ட ஒப்பீடு கிடைக்கவில்லை',
        'rainfall_analysis_unavailable': '📊 மழை பகுப்பாய்வு கிடைக்கவில்லை',
        'detailed_calc': '💡 விரிவான கணிப்பு பிரிவு பார்க்கவும்',
        'input_params': '📐 உள்ளீடு அளவுருக்கள்',
        'yield_calc': '🎯 விளைச்சல் கணிப்புகள்',
        'production_calc': '📦 உற்பத்தி கணிப்புகள்',
        'conversions': '💡 மாற்றங்கள்',
        'success': '✅ கணிப்பு வெற்றிகரமாக முடிந்தது!',
        'error': '❌ கணிப்பு பிழை: {error}',
        'check_inputs': '💡 உங்கள் உள்ளீடு மதிப்புகளை சரிபார்க்கவும்',
        'footer_title': '🌾 பயிர் விளைச்சல் கணிப்பான் v3.2',
        'footer_subtitle': 'மாவட்ட-வாரியாக பகுப்பாய்வு & மழை அறிவாற்றலுடன்',
        'footer_copyright': '© 2025 • மேம்பட்ட இயந்திர கற்றல் & தரவு அறிவியல் சக்தியளிக்கப்பட்டது'
    },
    'te': {
        'title': '🌾 పంట దిగుబడి అంచనా',
        'subtitle': 'జిల్లా & వర్షపాత అంతర్దృష్టులతో ఆధారిత వ్యవసాయ విశ్లేషణ',
        'enter_farm_details': '📝 ఫామ్ వివరాలు నమోదు చేయండి',
        'state': '🗺️ రాష్ట్రం ఎంచుకోండి',
        'district': '🏘️ జిల్లాను ఎంచుకోండి',
        'crop': '🌱 పంట ఎంచుకోండి',
        'season': '🌤️ సీజన్ ఎంచుకోండి',
        'area': '📐 ఫామ్ ఏరియా (ఎకరాలు)',
        'area_help': 'మొత్తం ఫామ్ ఏరియాను ఎకరాలలో నమోదు చేయండి',
        'year': '📅 పంట సంవత్సరం',
        'year_help': 'అంచనా కోసం సంవత్సరం ఎంచుకోండి',
        'rainfall': '🌧️ వార్షిక వర్షపాతం (మి.మి)',
        'rainfall_help': '{district} సగటు వర్షపాతం: {rainfall:.1f} మి.మి',
        'predict_btn': '🚀 ఇప్పుడు దిగుబడి అంచనా చేయండి',
        'historical_insights': '📊 చారిత్రక సమాచారం',
        'avg_yield': '📊 సగటు దిగుబడి',
        'max_yield': '🏆 గరిష్ఠ దిగుబడి',
        'records': '📝 రికార్డులు',
        'avg_rainfall': '🌧️ సగటు వర్షపాతం',
        'district_label': '📍 జిల్లా',
        'no_data': '{district}లో {crop}కి చారిత్రక డేటా లేదు',
        'no_load': '⚠️ చారిత్రక డేటా లోడ్ కాలేదు',
        'state_district': '{state} • {district} • {crop} • {season} • {year}',
        'area_card': '📐 ఏరియా',
        'yield_acre': '🎯 దిగుబడి/ఎకరా',
        'yield_ha': '🎯 దిగుబడి/హెక్టార్',
        'production': '📦 ఉత్పత్తి',
        'rainfall_card': '🌧️ వర్షపాతం',
        'production_summary': '📊 ఉత్పత్తి సారాంశం',
        'area_metric': 'ఏరియా',
        'hectares': 'హెక్టార్లు',
        'yield_acre_metric': 'దిగుబడి/ఎకరా',
        'production_metric': 'ఉత్పత్తి',
        'district_metric': 'జిల్లా',
        'detailed_analysis': '📈 వివరణాత్మక విశ్లేషణ & అంతర్దృష్టులు',
        'metrics_overview': 'ఉత్పత్తి మెట్రిక్స్ అవలోకనం',
        'comparison': '{district} - పోలిక',
        'higher': '✅ జిల్లా సగటు కంటే {pct:.1f}% ఎక్కువ',
        'lower': '⚠️ జిల్లా సగటు కంటే {pct:.1f}% తక్కువ',
        'equal': 'ℹ️ జిల్లా సగటుతో సమానం',
        'comp_unavailable': '📊 పోలిక అందుబాటులో లేదు',
        'district_comparison': '{state}లో జిల్లా-వారీగా పోలిక',
        'top_districts': '{crop}కి టాప్ 10 జిల్లాలు',
        'rainfall_vs_yield': '{state} - వర్షపాతం vs దిగుబడి',
        'district_comp_unavailable': '📊 జిల్లా పోలిక అందుబాటులో లేదు',
        'rainfall_analysis_unavailable': '📊 వర్షపాతం విశ్లేషణ అందుబాటులో లేదు',
        'detailed_calc': '💡 వివరణాత్మక లెక్కల బ్రేక్‌డౌన్ చూడండి',
        'input_params': '📐 ఇన్‌పుట్ పారామీటర్లు',
        'yield_calc': '🎯 దిగుబడి లెక్కలు',
        'production_calc': '📦 ఉత్పత్తి లెక్కలు',
        'conversions': '💡 మార్పిడులు',
        'success': '✅ అంచనా విజయవంతంగా పూర్తయింది!',
        'error': '❌ అంచనా లోపం: {error}',
        'check_inputs': '💡 మీ ఇన్‌పుట్ విలువలను తనిఖీ చేయండి',
        'footer_title': '🌾 పంట దిగుబడి అంచనా v3.2',
        'footer_subtitle': 'జిల్లా-వారీ విశ్లేషణ & వర్షపాతం బుద్ధిమత్తతో',
        'footer_copyright': '© 2025 • అధునాతన మెషిన్ లెర్నింగ్ & డేటా సైన్స్ ద్వారా సామర్థ్యం కల్పించబడింది'
    }
}

# Language selection - Top right corner
languages = ['English', 'हिंदी', 'ಕನ್ನಡ', 'தமிழ்', 'తెలుగు']
lang_display = ['English', 'Hindi', 'Kannada', 'Tamil', 'Telugu']
lang_codes = ['en', 'hi', 'kn', 'ta', 'te']

# Initialize session state for language
if 'selected_lang' not in st.session_state:
    st.session_state.selected_lang = 'en'

# Create language selector in top right
col_lang_left, col_lang_right = st.columns([6, 1])
with col_lang_right:
    selected_idx = st.selectbox(
        "🌐", 
        range(len(languages)), 
        index=lang_codes.index(st.session_state.selected_lang),
        format_func=lambda x: lang_display[x],
        key="lang_selector"
    )
    st.session_state.selected_lang = lang_codes[selected_idx]
selected_lang = st.session_state.selected_lang
t = TRANSLATIONS[selected_lang]

# Helper function for charts
def create_styled_chart(figsize=(10, 6)):
    """Create matplotlib figure with transparent styling"""
    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    ax.set_facecolor('none')
    return fig, ax

# Helper function for formatting values
def format_value(value):
    """Format value for display"""
    if value > 100:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"

# Cache functions
@st.cache_resource
def load_models():
    """Load models and encoders"""
    try:
        model = joblib.load('model.pkl')
        le_state = joblib.load('state_encoder.pkl')
        le_district = joblib.load('district_encoder.pkl')
        le_crop = joblib.load('crop_encoder.pkl')
        le_season = joblib.load('season_encoder.pkl')
        return model, le_state, le_district, le_crop, le_season
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("💡 Run train_model.py first to generate model files")
        st.stop()

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('agridata.csv')
        
        # --- CRITICAL FIX: STRIP HEADER SPACES ---
        df.columns = df.columns.str.strip()
        
        # --- PRE-PROCESS RAINFALL ---
        if 'Annual_Rainfall' in df.columns:
            # Fill missing values: District Mean -> State Mean -> Global Mean
            df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(
                df.groupby(['State', 'District'])['Annual_Rainfall'].transform('mean')
            )
            df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(
                df.groupby('State')['Annual_Rainfall'].transform('mean')
            )
            df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(df['Annual_Rainfall'].mean())
        
        # Derived columns (Assuming 'Yield' is kg/ha or t/ha - standardized here)
        # Using specific calculations to match your training logic
        if 'Yield' in df.columns:
            df['Yield_Q_Acre'] = df['Yield'] / 247.1 # Adjust logic if Yield is different unit
        else:
            # Calculate fresh if needed
            df['Yield_Q_Acre'] = (df['Production'] * 1000 / df['Area']) / 247.1

        df['Yield_Q_Ha'] = df['Yield_Q_Acre'] * 2.471
        df['Area_Acres'] = df['Area'] * 2.471 # Assuming Area is Hectares
        
        return df
    except FileNotFoundError:
        st.error("❌ Dataset 'agridata.csv' not found!")
        st.info("💡 Please place agridata.csv in the same directory")
        st.stop()

# Initialize
try:
    model, le_state, le_district, le_crop, le_season = load_models()
    df = load_data()
except Exception as e:
    st.error(f"❌ Initialization error: {e}")
    st.stop()

# Title
st.markdown(f"""
<div class="header-container">
    <div class="main-title">{t['title']}</div>
    <div class="subtitle">{t['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"#### {t['enter_farm_details']}")
    
    # State selection
    state = st.selectbox(t['state'], sorted(le_state.classes_), key="state_select")
    
    # Dynamic district selection
    try:
        districts_in_state = sorted(df[df['State'] == state]['District'].unique())
        if len(districts_in_state) == 0:
            districts_in_state = sorted(le_district.classes_)
    except:
        districts_in_state = sorted(le_district.classes_)
    
    district = st.selectbox(t['district'], districts_in_state, key="district_select")
    
    # Crop and Season
    crop = st.selectbox(t['crop'], sorted(le_crop.classes_), key="crop_select")
    season = st.selectbox(t['season'], sorted(le_season.classes_), key="season_select")
    
    # Area input
    area_acres = st.number_input(
        t['area'],
        min_value=0.1,
        max_value=250000.0,
        value=1000.0,
        step=10.0,
        help=t['area_help']
    )
    
    # Year input
    crop_year = st.number_input(
        t['year'],
        min_value=2000,
        max_value=2030,
        value=2024,
        help=t['year_help']
    )
    
    # --- FIXED RAINFALL AUTO-FETCH ---
    try:
        # Fetch specific rainfall for this State + District
        district_rain_data = df[
            (df['State'] == state) & 
            (df['District'] == district)
        ]
        
        if not district_rain_data.empty and 'Annual_Rainfall' in district_rain_data.columns:
            fetched_rainfall = district_rain_data['Annual_Rainfall'].mean()
        else:
            # Fallback to state average
            state_rain_data = df[df['State'] == state]
            if not state_rain_data.empty:
                fetched_rainfall = state_rain_data['Annual_Rainfall'].mean()
            else:
                fetched_rainfall = 1000.0
                
        default_rainfall = float(fetched_rainfall)
        
    except Exception as e:
        default_rainfall = 1000.0
    
    if pd.isna(default_rainfall):
        default_rainfall = 1000.0

    annual_rainfall = st.number_input(
        f"{t['rainfall']} - Avg: {default_rainfall:.1f} mm",
        min_value=0.0,
        max_value=5000.0,
        value=default_rainfall,
        step=10.0,
        help=t['rainfall_help'].format(district=district, rainfall=default_rainfall)
    )
    
    area_hectares = area_acres * 0.4047
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(t['predict_btn'], use_container_width=True)

with col2:
    st.markdown(f"#### {t['historical_insights']}")
    
    try:
        hist_data = df[(df['State'] == state) & (df['District'] == district) & (df['Crop'] == crop)]
        
        if not hist_data.empty:
            avg_yield = hist_data['Yield_Q_Acre'].mean()
            max_yield = hist_data['Yield_Q_Acre'].max()
            records = len(hist_data)
            avg_rainfall_hist = hist_data['Annual_Rainfall'].mean() if 'Annual_Rainfall' in hist_data.columns else 0
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(t['avg_yield'], f"{avg_yield:.2f} q/acre")
            col_m2.metric(t['max_yield'], f"{max_yield:.2f} q/acre")
            col_m3.metric(t['records'], f"{records:,}")
            
            col_m4, col_m5 = st.columns(2)
            col_m4.metric(t['avg_rainfall'], f"{avg_rainfall_hist:.0f} mm")
            col_m5.metric(t['district_label'], district)
            
            # Trend chart
            yearly_data = hist_data.groupby('Crop_Year')['Yield_Q_Acre'].mean().reset_index()
            
            if len(yearly_data) > 1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig, ax = create_styled_chart(figsize=(8, 4))
                
                ax.fill_between(yearly_data['Crop_Year'], yearly_data['Yield_Q_Acre'], 
                              alpha=0.3, color='#4ecca3')
                ax.plot(yearly_data['Crop_Year'], yearly_data['Yield_Q_Acre'],
                        linewidth=2.5, marker='o', markersize=6, color='#4ecca3',
                        markeredgecolor='white', markeredgewidth=1.5)
                
                ax.set_xlabel('Year', fontsize=10, fontweight='bold', color='white')
                ax.set_ylabel('Yield (q/acre)', fontsize=10, fontweight='bold', color='white')
                ax.set_title(f'{crop} Yield Trend - {district}', fontsize=11, fontweight='bold', color='white', pad=12)
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.tick_params(colors='white', labelsize=8)
                
                for spine in ax.spines.values():
                    spine.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                    spine.set_linewidth(1.5)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(t['no_data'].format(crop=crop, district=district))
    except Exception as e:
        st.warning(t['no_load'])
    
# Prediction Section
if predict_btn:
    try:
        # Encode inputs
        state_enc = le_state.transform([state])[0]
        district_enc = le_district.transform([district])[0]
        crop_enc = le_crop.transform([crop])[0]
        season_enc = le_season.transform([season])[0]
        
        # Create input array (7 features)
        input_data = np.array([[state_enc, district_enc, crop_enc, season_enc,
                              area_acres, crop_year, annual_rainfall]])
        
        # Make prediction
        yield_q_acre = model.predict(input_data)[0]
        yield_kg_ha = yield_q_acre * 247.1
        yield_q_ha = yield_kg_ha / 100
        total_production_tonnes = yield_kg_ha * area_hectares / 1000
        total_production_quintals = total_production_tonnes * 10
        
        # Progress animation
        with st.spinner(t['analyzing']):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.008)
                progress_bar.progress(i + 1)
            progress_bar.empty()
        
        st.markdown("---")
        st.markdown(f"### 📍 {t['state_district'].format(state=state, district=district, crop=crop, season=season, year=crop_year)}")
        
        # Prediction cards
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        
        # Card 1: Area
        with col_r1:
            area_display = format_value(area_acres)
            st.markdown(f"""
            <div class="prediction-card">
                <div class="card-label">{t['area_card']}</div>
                <div class="card-value">{area_display}</div>
                <div class="card-unit">acres</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 2: Yield/Acre
        with col_r2:
            yield_acre_display = format_value(yield_q_acre)
            st.markdown(f"""
            <div class="prediction-card">
                <div class="card-label">{t['yield_acre']}</div>
                <div class="card-value">{yield_acre_display}</div>
                <div class="card-unit">q/acre</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 3: Yield/Ha
        with col_r3:
            yield_ha_display = format_value(yield_q_ha)
            st.markdown(f"""
            <div class="prediction-card">
                <div class="card-label">{t['yield_ha']}</div>
                <div class="card-value">{yield_ha_display}</div>
                <div class="card-unit">q/ha</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 4: Production
        with col_r4:
            production_display = format_value(total_production_quintals)
            st.markdown(f"""
            <div class="prediction-card">
                <div class="card-label">{t['production']}</div>
                <div class="card-value">{production_display}</div>
                <div class="card-unit">quintals</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Card 5: Rainfall
        with col_r5:
            rainfall_display = format_value(annual_rainfall)
            st.markdown(f"""
            <div class="prediction-card">
                <div class="card-label">{t['rainfall_card']}</div>
                <div class="card-value">{rainfall_display}</div>
                <div class="card-unit">mm</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("---")
        st.subheader(t['production_summary'])
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        col_s1.metric(t['area_metric'], f"{area_acres:,.0f} acres")
        col_s2.metric(t['hectares'], f"{area_hectares:.2f} ha")
        col_s3.metric(t['yield_acre_metric'], f"{yield_q_acre:.2f} q")
        col_s4.metric(t['production_metric'], f"{total_production_tonnes:,.2f} t")
        col_s5.metric(t['district_metric'], district)
        
        # Analysis Charts
        st.markdown("---")
        st.subheader(t['detailed_analysis'])
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            fig, ax = create_styled_chart()
            
            categories = ['Yield\n/Acre', 'Yield\n/Ha', 'Area\n(ha/10)', 'Production\n(t/10)', 'Rainfall\n(mm/100)']
            values = [yield_q_acre, yield_q_ha, area_hectares/10, total_production_tonnes/10, annual_rainfall/100]
            colors_bar = ['#4ecca3', '#2d6a4f', '#1b4332', '#0f5f3f', '#4facfe']
            
            bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=2)
            
            for bar, val, real_val in zip(bars, values, [yield_q_acre, yield_q_ha, area_hectares, total_production_tonnes, annual_rainfall]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                        f'{real_val:.1f}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=9)
            
            ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold', color='white')
            ax.set_title(t['metrics_overview'], fontsize=13, fontweight='bold', color='white', pad=15)
            ax.tick_params(colors='white', labelsize=9)
            ax.grid(True, alpha=0.2, axis='y')
            
            for spine in ax.spines.values():
                spine.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                spine.set_linewidth(1.5)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_v2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            try:
                if not hist_data.empty:
                    avg_yield_comp = hist_data['Yield_Q_Acre'].mean()
                    diff = yield_q_acre - avg_yield_comp
                    diff_pct = (diff / avg_yield_comp * 100) if avg_yield_comp > 0 else 0
                    
                    fig, ax = create_styled_chart()
                    
                    categories = ['Historical\nAverage', 'Your\nPrediction']
                    values = [avg_yield_comp, yield_q_acre]
                    colors = ['#2d6a4f', '#4ecca3']
                    
                    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2, width=0.6)
                    
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                               f'{val:.2f}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=11)
                    
                    ax.set_ylabel('Yield (q/acre)', fontsize=11, fontweight='bold', color='white')
                    ax.set_title(t['comparison'].format(district=district), fontsize=13, fontweight='bold', color='white', pad=15)
                    ax.tick_params(colors='white', labelsize=9)
                    ax.grid(True, alpha=0.2, axis='y')
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    
                    if diff_pct > 0:
                        st.success(t['higher'].format(pct=diff_pct))
                    elif diff_pct < 0:
                        st.warning(t['lower'].format(pct=diff_pct))
                    else:
                        st.info(t['equal'])
            except:
                st.info(t['comp_unavailable'])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # District Comparison
        st.markdown("---")
        st.subheader(t['district_comparison'].format(state=state))
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                state_data = df[(df['State'] == state) & (df['Crop'] == crop)]
                if not state_data.empty:
                    district_avg = state_data.groupby('District')['Yield_Q_Acre'].mean().sort_values(ascending=False).head(10)
                    
                    fig, ax = create_styled_chart()
                    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(district_avg)))
                    bars = ax.barh(range(len(district_avg)), district_avg.values, color=colors, edgecolor='white', linewidth=1.5)
                    
                    if district in district_avg.index:
                        idx = list(district_avg.index).index(district)
                        bars[idx].set_edgecolor('#f093fb')
                        bars[idx].set_linewidth(3)
                    
                    for i, val in enumerate(district_avg.values):
                        ax.text(val + max(district_avg.values)*0.02, i, f'{val:.2f}', va='center', fontweight='bold', color='white', fontsize=9)
                    
                    ax.set_yticks(range(len(district_avg)))
                    ax.set_yticklabels(district_avg.index, fontsize=10)
                    ax.set_xlabel('Average Yield (q/acre)', fontsize=11, fontweight='bold', color='white')
                    ax.set_title(t['top_districts'].format(crop=crop), fontsize=13, fontweight='bold', color='white', pad=15)
                    ax.grid(True, alpha=0.2, axis='x')
                    ax.tick_params(colors='white', labelsize=9)
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            except:
                st.info(t['district_comp_unavailable'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_d2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            try:
                if not state_data.empty and 'Annual_Rainfall' in state_data.columns:
                    sample = state_data.sample(min(200, len(state_data)))
                    
                    fig, ax = create_styled_chart()
                    scatter = ax.scatter(sample['Annual_Rainfall'], sample['Yield_Q_Acre'],
                                       c=sample['Yield_Q_Acre'], cmap='plasma', s=80, alpha=0.6, edgecolors='white', linewidth=1)
                    
                    ax.scatter([annual_rainfall], [yield_q_acre], color='#f093fb', s=300, marker='*', edgecolors='white', linewidth=2, zorder=5, label='Your Prediction')
                    
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Yield (q/acre)', rotation=270, labelpad=20, color='white', fontweight='bold')
                    cbar.ax.tick_params(colors='white')
                    cbar.outline.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                    
                    ax.set_xlabel('Annual Rainfall (mm)', fontsize=11, fontweight='bold', color='white')
                    ax.set_ylabel('Yield (q/acre)', fontsize=11, fontweight='bold', color='white')
                    ax.set_title(t['rainfall_vs_yield'].format(state=state), fontsize=13, fontweight='bold', color='white', pad=15)
                    ax.legend(loc='best', framealpha=0.3, facecolor='black', edgecolor='white')
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(colors='white', labelsize=9)
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor((1.0, 1.0, 1.0, 0.3))
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            except:
                st.info(t['rainfall_analysis_unavailable'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Calculation
        with st.expander(t['detailed_calc']):
            col_calc1, col_calc2 = st.columns(2)
            
            with col_calc1:
                st.write(f"**{t['input_params']}**")
                st.write(f"• State: `{state}`")
                st.write(f"• District: `{district}`")
                st.write(f"• Crop: `{crop}`")
                st.write(f"• Season: `{season}`")
                st.write(f"• Year: `{crop_year}`")
                st.write(f"• Area: `{area_acres:,.0f} acres = {area_hectares:.2f} ha`")
                st.write(f"• Rainfall: `{annual_rainfall:.0f} mm`")
                st.write("")
                st.write(f"**{t['yield_calc']}**")
                st.write(f"• Predicted: `{yield_q_acre:.2f} q/acre`")
                st.write(f"• Per Ha: `{yield_q_ha:.2f} q/ha`")
                st.write(f"• kg/ha: `{yield_kg_ha:.2f} kg/ha`")
            
            with col_calc2:
                st.write(f"**{t['production_calc']}**")
                st.write(f"• Total (Quintals):")
                st.write(f"  `{yield_q_ha:.2f} × {area_hectares:.2f}`")
                st.write(f"  `= {total_production_quintals:,.2f} q`")
                st.write("")
                st.write(f"• Total (Tonnes):")
                st.write(f"  `{total_production_tonnes:,.2f} ÷ 10`")
                st.write(f"  `= {total_production_tonnes:,.2f} t`")
                st.write("")
                st.write(f"**{t['conversions']}**")
                st.write("• 1 acre = 0.4047 ha")
                st.write("• 1 quintal = 100 kg")
                st.write("• 1 tonne = 10 quintals")
        
        st.success(t['success'])
        
    except Exception as e:
        st.error(t['error'].format(error=str(e)))
        st.info(t['check_inputs'])

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 2rem 0;'>
    <p style='color: white; font-size: 1.2rem; font-weight: 700; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);'>
        {t['footer_title']}
    </p>
    <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-top: 0.5rem;'>
        {t['footer_subtitle']}
    </p>
    <p style='color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin-top: 0.3rem;'>
        {t['footer_copyright']}
    </p>
</div>
""", unsafe_allow_html=True)
