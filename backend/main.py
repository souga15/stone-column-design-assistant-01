"""
Stone Column Design Assistant V6 - Complete Professional Edition
Advanced AI-powered geotechnical design tool with comprehensive analytics
Uses 2-output model (Ultimate Stress, Service Load) with computed Factor of Safety
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from datetime import datetime

st.set_page_config(page_title="Stone Column Design Assistant V6", page_icon="🗿", layout="wide")

# STYLING
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: gradient-animation 3s ease infinite; background-size: 200% 200%;
    }
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main-header h1 { color: white; font-size: 2.8rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: rgba(255,255,255,0.95); font-size: 1.2rem; margin-top: 0.5rem; }
    .stMetric { background: linear-gradient(135deg, #1e2129 0%, #2d3139 100%); padding: 1.5rem;
        border-radius: 12px; border-left: 4px solid #667eea; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.3s ease; }
    .stMetric:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(102,126,234,0.3); }
    h2, h3 { color: #667eea; font-weight: 600; padding-bottom: 0.5rem; border-bottom: 2px solid rgba(102,126,234,0.3); margin-top: 2rem; }
    .info-card { background: linear-gradient(135deg, #2d3139 0%, #3d4149 100%); padding: 1.5rem;
        border-radius: 10px; border-left: 4px solid #764ba2; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL AND SCALERS
@st.cache_resource
def load_all():
    try:
        model = tf.keras.models.load_model("stone_column_ann_model.h5", compile=False)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        return None, None, None

model, scaler_X, scaler_y = load_all()

# PARAMETERS
PARAM_RANGES = {
    'cu': (5.0, 40.0, 15.0), 'D': (0.06, 0.8, 0.4), 'L': (0.7, 12.0, 6.0),
    'sD': (2.0, 4.0, 2.5), 'Eenc': (0.0, 20.0, 0.0)
}

PARAM_INFO = {
    'cu': {'name': 'Undrained Shear Strength', 'unit': 'kPa'},
    'D': {'name': 'Column Diameter', 'unit': 'm'},
    'L': {'name': 'Column Length', 'unit': 'm'},
    'sD': {'name': 'Spacing Ratio (s/D)', 'unit': '-'},
    'Eenc': {'name': 'Encasement Stiffness', 'unit': 'kN/m'}
}

# FUNCTIONS
def predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc):
    x = np.array([[cu, D, L, sD, Eenc]], dtype=np.float32)
    x_scaled = scaler_X.transform(x)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    
    sigma, P10 = pred[0], pred[1]
    FS = compute_FS(sigma, P10, cu, D, L, sD)
    return sigma, P10, FS

def compute_FS(sigma, P10, cu, D, L, sD):
    if P10 <= 0 or sigma <= 0:
        return 0.0
    FS = sigma / P10
    corr = 1.0
    if cu < 10: corr *= (0.7 + 0.03 * cu)
    if D < 0.25: corr *= (0.8 + 0.8 * D)
    if sD < 2.5: corr *= (0.85 + 0.06 * sD)
    if L < 3.0: corr *= (0.75 + 0.083 * L)
    return max(0.5, min(FS * corr, 5.0))

def calc_derived(cu, D, L, sD, sigma, P10):
    A_col = np.pi * (D/2)**2
    return {
        'spacing': sD * D, 'slenderness': L/D, 'area_repl': 1/(sD**2),
        'improv_factor': sigma/cu if cu > 0 else 0, 'col_area': A_col,
        'load_kN': P10 * A_col, 'load_per_len': (P10 * A_col)/L if L > 0 else 0
    }

def validate_design(cu, D, L, sD):
    warnings, reliable = [], True
    if cu < 10: warnings.append(f"⚠️ Weak soil (cu={cu:.1f} kPa)"); reliable = False
    if D < 0.25: warnings.append(f"⚠️ Small diameter (D={D:.2f} m)"); reliable = False
    if L < 3.0: warnings.append(f"⚠️ Short column (L={L:.1f} m)"); reliable = False
    if sD < 2.5: warnings.append(f"⚠️ Tight spacing (s/D={sD:.1f})"); reliable = False
    if L/D > 20: warnings.append(f"⚠️ High slenderness (L/D={L/D:.1f})")
    return reliable, warnings

def assess_safety(FS, reliable):
    if not reliable: return "UNRELIABLE", "Outside validated range", "error"
    if FS >= 2.5: return "Excellent", "Exceeds requirements", "success"
    elif FS >= 2.0: return "Adequate", "Meets requirements", "success"
    elif FS >= 1.5: return "Marginal", "Consider optimization", "warning"
    else: return "Insufficient", "Revision required", "error"

# HEADER
st.markdown('<div class="main-header"><h1>Stone Column Design Assistant V6</h1>'
           '<p>AI-Powered Geotechnical Design with Comprehensive Analytics</p></div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("Design Parameters")
    st.markdown("---")
    cu = st.number_input(f"{PARAM_INFO['cu']['name']} ({PARAM_INFO['cu']['unit']})", 
                         PARAM_RANGES['cu'][0], PARAM_RANGES['cu'][1], PARAM_RANGES['cu'][2], 0.5)
    D = st.number_input(f"{PARAM_INFO['D']['name']} ({PARAM_INFO['D']['unit']})", 
                        PARAM_RANGES['D'][0], PARAM_RANGES['D'][1], PARAM_RANGES['D'][2], 0.01, format="%.2f")
    L = st.number_input(f"{PARAM_INFO['L']['name']} ({PARAM_INFO['L']['unit']})", 
                        PARAM_RANGES['L'][0], PARAM_RANGES['L'][1], PARAM_RANGES['L'][2], 0.1, format="%.1f")
    sD = st.number_input(f"{PARAM_INFO['sD']['name']} ({PARAM_INFO['sD']['unit']})", 
                         PARAM_RANGES['sD'][0], PARAM_RANGES['sD'][1], PARAM_RANGES['sD'][2], 0.1, format="%.1f")
    Eenc = st.number_input(f"{PARAM_INFO['Eenc']['name']} ({PARAM_INFO['Eenc']['unit']})", 
                           PARAM_RANGES['Eenc'][0], PARAM_RANGES['Eenc'][1], PARAM_RANGES['Eenc'][2], 0.5, format="%.1f")
    st.info(f"**Spacing:** {sD * D:.2f} m")
    st.markdown("---")
    st.header("Analysis Options")
    sens = st.checkbox("Sensitivity Analysis", True)
    heat = st.checkbox("Interaction Heatmap", True)
    surf3d = st.checkbox("3D Surface Plot", True)

# PREDICT
if model is None:
    st.warning("Model not loaded")
    st.stop()

inp = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}
sigma, P10, FS = predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc)
der = calc_derived(cu, D, L, sD, sigma, P10)
reliable, warns = validate_design(cu, D, L, sD)

# RESULTS
st.header("Prediction Results")
if warns:
    st.error("**Reliability Warnings:**")
    for w in warns: st.write(w)
    st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultimate Stress", f"{sigma:.2f} kPa", help="AI predicted")
c2.metric("Service Load", f"{P10:.2f} kPa", help="AI predicted")
c3.metric("Factor of Safety", f"{FS:.2f}", delta="Safe" if FS >= 2.0 else "Low",
         delta_color="normal" if FS >= 2.0 else "inverse", help="Computed: σ/P10")
c4.metric("Slenderness", f"{der['slenderness']:.1f}")

st.subheader("Design Assessment")
status, msg, col = assess_safety(FS, reliable)
if col == "success": st.success(f"**{status}:** {msg}")
elif col == "warning": st.warning(f"**{status}:** {msg}")
else: st.error(f"**{status}:** {msg}")

st.subheader("Design Information")
c1, c2 = st.columns(2)
c1.info(f"""**Column Config:** Type: {'Encased' if Eenc > 0 else 'Unencased'}\n
Area Replacement: {der['area_repl']:.3f} | Spacing: {der['spacing']:.2f} m\n
Slenderness: {der['slenderness']:.1f} | Area: {der['col_area']:.4f} m²""")

eff = "High" if der['improv_factor'] > 3 else "Moderate" if der['improv_factor'] > 2 else "Low"
c2.info(f"""**Performance:** Improvement: {der['improv_factor']:.2f}x\n
Service Load: {der['load_kN']:.2f} kN | Load/Length: {der['load_per_len']:.2f} kN/m\n
Efficiency: {eff} | Reliability: {'OK' if reliable else 'UNRELIABLE'}""")

st.markdown("---")

# SENSITIVITY
if sens:
    st.header("Sensitivity Analysis")
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    sel = st.selectbox("Parameter:", list(opts.keys()))
    key = opts[sel]
    
    vals = np.linspace(PARAM_RANGES[key][0], PARAM_RANGES[key][1], 60)
    preds = []
    for v in vals:
        t = inp.copy(); t[key] = v
        x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
        xs = scaler_X.transform(x)
        ps = model.predict(xs, verbose=0)[0]
        p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
        fs = compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        preds.append([p[0], p[1], fs])
    preds = np.array(preds)
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Ultimate Stress", "Service Load", "Factor of Safety"])
    colors = ['#667eea', '#26c6da', '#ffa726']
    for i in range(3):
        fig.add_trace(go.Scatter(x=vals, y=preds[:,i], mode='lines',
                                line=dict(color=colors[i], width=3), fill='tozeroy'), row=1, col=i+1)
        fig.add_vline(x=inp[key], line_dash="dash", line_color="red", row=1, col=i+1)
    fig.add_hline(y=2.0, line_dash="dot", line_color="green", row=1, col=3)
    fig.update_layout(height=450, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', 
                     paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Min σ", f"{preds[:,0].min():.2f} kPa"); c1.metric("Max σ", f"{preds[:,0].max():.2f} kPa")
    c2.metric("Min P10", f"{preds[:,1].min():.2f} kPa"); c2.metric("Max P10", f"{preds[:,1].max():.2f} kPa")
    c3.metric("Min FS", f"{preds[:,2].min():.2f}"); c3.metric("Max FS", f"{preds[:,2].max():.2f}")
    st.markdown("---")

# HEATMAP
if heat:
    st.header("Parameter Interaction")
    c1, c2, c3 = st.columns(3)
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    p1n = c1.selectbox("X:", list(opts.keys()), key='h1')
    p2n = c2.selectbox("Y:", list(opts.keys()), index=1, key='h2')
    outn = c3.selectbox("Output:", ["Ultimate Stress", "Service Load", "Factor of Safety"])
    
    if p1n != p2n:
        p1k, p2k = opts[p1n], opts[p2n]
        res = 40
        p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
        p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
        Z = np.zeros((len(p2v), len(p1v)))
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)
        
        for i, v2 in enumerate(p2v):
            for j, v1 in enumerate(p1v):
                t = inp.copy(); t[p1k], t[p2k] = v1, v2
                x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
                xs = scaler_X.transform(x)
                ps = model.predict(xs, verbose=0)[0]
                p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
                Z[i,j] = p[oidx] if oidx < 2 else compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        
        fig = go.Figure(go.Heatmap(z=Z, x=p1v, y=p2v, colorscale='Plasma'))
        fig.add_scatter(x=[inp[p1k]], y=[inp[p2k]], mode='markers+text',
                       marker=dict(size=20, color='red', symbol='star', line=dict(width=3, color='white')),
                       text=['Current'], textposition='top center')
        fig.update_layout(title=f"{outn} Interaction", xaxis_title=p1n, yaxis_title=p2n,
                         height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        oi = np.unravel_index(np.argmax(Z), Z.shape)
        st.success(f"Optimal: {p1n}={p1v[oi[1]]:.2f}, {p2n}={p2v[oi[0]]:.2f}, {outn}={Z[oi]:.2f}")
    st.markdown("---")

# 3D SURFACE
if surf3d:
    st.header("3D Surface")
    c1, c2, c3 = st.columns(3)
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    p1n = c1.selectbox("X:", list(opts.keys()), key='3d1')
    p2n = c2.selectbox("Y:", list(opts.keys()), index=1, key='3d2')
    outn = c3.selectbox("Z:", ["Ultimate Stress", "Service Load", "Factor of Safety"])
    
    if p1n != p2n:
        res = st.slider("Resolution", 15, 30, 20)
        p1k, p2k = opts[p1n], opts[p2n]
        p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
        p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
        P1, P2 = np.meshgrid(p1v, p2v)
        Z = np.zeros_like(P1)
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)
        
        for i in range(P1.shape[0]):
            for j in range(P1.shape[1]):
                t = inp.copy(); t[p1k], t[p2k] = P1[i,j], P2[i,j]
                x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
                xs = scaler_X.transform(x)
                ps = model.predict(xs, verbose=0)[0]
                p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
                Z[i,j] = p[oidx] if oidx < 2 else compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        
        fig = go.Figure(go.Surface(z=Z, x=p1v, y=p2v, colorscale='Viridis'))
        fig.update_layout(scene=dict(xaxis_title=p1n, yaxis_title=p2n, zaxis_title=outn),
                         height=700, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# EXPORT
st.header("Export Results")
with st.expander("Download"):
    df = pd.DataFrame({
        'Parameter': [PARAM_INFO[k]['name'] for k in PARAM_INFO] + 
                     ['', 'Ultimate Stress', 'Service Load', 'Factor of Safety'],
        'Value': [cu, D, L, sD, Eenc, '', sigma, P10, FS],
        'Unit': [PARAM_INFO[k]['unit'] for k in PARAM_INFO] + ['', 'kPa', 'kPa', '-']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), 
                      f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

# RECOMMENDATIONS
st.header("Recommendations")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Structural")
    if FS < 1.5:
        st.error("**Critical Issues**\n- Increase diameter/length\n- Reduce spacing\n- Add encasement")
    elif FS < 2.0:
        st.warning("**Optimization Suggested**\n- Adjust dimensions\n- Reduce spacing 10-15%")
    else:
        st.success("**Design Sound**\n- Meets safety criteria\n- Can optimize for cost")

with c2:
    st.subheader("Economic")
    cost = (D**2) * L / (sD**2)
    if cost > 2.0:
        st.warning("**High Material**\n- Reduce diameter if FS allows\n- Increase spacing")
    else:
        st.success("**Cost-Efficient**\n- Optimized material usage")

# SUMMARY
st.header("Final Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Status", status); c1.metric("FS", f"{FS:.2f}"); c1.metric("Type", "Encased" if Eenc > 0 else "Unencased")
c2.metric("σ", f"{sigma:.1f} kPa"); c2.metric("P10", f"{P10:.1f} kPa"); c2.metric("Improvement", f"{der['improv_factor']:.2f}x")
c3.metric("Spacing", f"{der['spacing']:.2f} m"); c3.metric("L/D", f"{der['slenderness']:.1f}"); c3.metric("Area Repl", f"{der['area_repl']:.3f}")

# FOOTER
st.markdown("---")
st.markdown('<div class="info-card"><b>⚠ Disclaimer</b><br>AI-assisted preliminary design only. '
           'Final design must be verified by qualified engineers using site data and design codes.</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; opacity:0.7; margin-top:2rem;">Stone Column Design Assistant V6 © 2026<br>'

           'FS computed using engineering formulas: σ/P10 with correction factors</div>', unsafe_allow_html=True)
