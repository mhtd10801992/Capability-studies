import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cp & Cpk Calculator", layout="centered")
st.markdown('<h1 style="font-size: 21pt; text-align: center;">Engineering</h1>', unsafe_allow_html=True)
st.title("Cp & Cpk Calculator")

st.markdown("""
Enter your measurement data, USL (Upper Spec Limit), and LSL (Lower Spec Limit) below. The app will calculate Cp and Cpk and display a live histogram.
""")

data_input = st.text_area("Enter measurement data (comma or space separated):", "")
lsl = st.number_input("Lower Spec Limit (LSL):", value=0.0, format="%f")
usl = st.number_input("Upper Spec Limit (USL):", value=1.0, format="%f")
target = st.number_input("Target Value:", value=float((usl + lsl) / 2), format="%f")


# Data validatio`n and parsing
def parse_data(input_str):
    try:
        data = [float(x) for x in input_str.replace(",", " ").split() if x.strip()]
        return np.array(data)
    except Exception:
        return np.array([])

data = parse_data(data_input)

if data.size > 1 and usl > lsl:
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    std_from_target = np.sqrt(np.mean((data - target) ** 2))
    st.write(f"**Std Dev from Target:** {std_from_target:.4f}")
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpu = (usl - mean) / (3 * std) if std > 0 else np.nan
    cpl = (mean - lsl) / (3 * std) if std > 0 else np.nan
    cpk = min(cpu, cpl)

    st.subheader("Results")
    st.write(f"**Mean:** {mean:.4f}")
    st.write(f"**Std Dev:** {std:.4f}")
    st.write(f"**Cp:** {cp:.4f}")
    st.write(f"**Cpk:** {cpk:.4f}")
    st.write(f"**Std Dev from Target:** {std_from_target:.4f}")


    fig, ax = plt.subplots(figsize=(8, 5))
    # Histogram as frequency (not density)
    n, bins, patches = ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=False, label='Data Histogram')
    # Bell curve (scaled to max frequency bar), centered at sample mean and std
    x = np.linspace(bins[0], bins[-1], 200)
    # The bell curve is always centered at the sample mean and uses the sample std,
    # so it will visually show the mean shift (if any) relative to the spec limits.
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    y_scaled = y * (max(n) / max(y)) if max(y) > 0 else y  # Scale bell curve to max bar height
    ax.plot(x, y_scaled, color='orange', linewidth=2, label='Normal Distribution (mean, std)')
    # Spec limits and mean
    ax.axvline(lsl, color='red', linestyle='dashed', linewidth=2, label='LSL')
    ax.axvline(usl, color='green', linestyle='dashed', linewidth=2, label='USL')
    ax.axvline(mean, color='blue', linestyle='solid', linewidth=2, label='Mean')
    # Place legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Frequency')
    ax.set_title('Measurement Histogram with Scaled Normal Curve')
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
     # Scatter plot
    st.subheader("Scatter Plot of Measurements")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.scatter(range(1, len(data) + 1), data, color='purple', alpha=0.7, label='Measurements')
    ax2.axhline(mean, color='blue', linestyle='solid', linewidth=2, label='Mean')
    ax2.axhline(target, color='orange', linestyle='dashed', linewidth=2, label='Target')
    ax2.axhline(lsl, color='red', linestyle='dashed', linewidth=2, label='LSL')
    ax2.axhline(usl, color='green', linestyle='dashed', linewidth=2, label='USL')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Measurement Value')
    ax2.set_title('Scatter Plot of Measurements')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)
else:
    st.info("Please enter at least two data points and ensure USL > LSL.")


