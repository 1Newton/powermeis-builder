import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import os
from PIL import Image
from geometrias import PrismaTriangular, CuboCoreShell, CilindroEnterrado, EsferaCoreShell, EsferoideSubstrato, Cubo, IRVMMatrix, CuboMulticamada, StackBuilderMatrix, ImportedMatrix, AdvancedStackMatrix, MembranaPorosa
# Page Configuration
st.set_page_config(page_title="PowerMEIS Matrix Builder", layout="wide", page_icon="⚛️")

st.title("PowerMEIS 3D Matrix Builder")
st.markdown("Create, visualize, and export geometries for the PowerMEIS simulator.")

# ==========================================
# BRANDING & CREDITS (SIDEBAR)
# ==========================================
if os.path.exists("logo_lab.png"):
    st.sidebar.image("logo_lab.png", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚛️ PowerMEIS Matrix Builder")
st.sidebar.markdown("**Developed by D'Newton**")
st.sidebar.markdown("🏛️ *Laboratório de Implantação Iônica (LII) - UFRGS*")
st.sidebar.markdown("🔗 [PowerMEIS Official Website](http://tars.if.ufrgs.br/)")
st.sidebar.markdown("---")

# ==========================================
# GEOMETRY CONFIGURATION
# ==========================================
st.sidebar.header("Geometry Settings")
options = [
    "IRVM (Image Reconstructed Voxel Model)",
    "Porous 2D Membrane",      # <-- Adicione o (MoS2) aqui
    "Universal Stack Builder",
    "Advanced Stack Builder",
    "Import & View (.mtx)",
    "Simple Cube (with vacuum padding)",
    "Cube with Top Film", 
    "Triangular Prism", 
    "Core/Shell Cube", 
    "Buried Cylinder", 
    "Core/Shell Sphere", 
    "Spheroid on Substrate"
]
geometry_type = st.sidebar.selectbox("Choose geometry type:", options)

# ==========================================
# MEMÓRIA (SESSION STATE)
# ==========================================
# Garante que a matriz não suma quando o usuário digitar o nome do arquivo
if "matriz_obj" not in st.session_state:
    st.session_state.matriz_obj = None
    st.session_state.last_geometry = geometry_type

# Se o usuário mudar o tipo de geometria no menu, limpa a tela anterior
if st.session_state.last_geometry != geometry_type:
    st.session_state.matriz_obj = None
    st.session_state.last_geometry = geometry_type


def st_hsv_segmentation(img_rgb, prefix):
    st.markdown(f"### {prefix} Segmentation (HSV & Morph)")
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        h_range = st.slider(f"Hue (H) - {prefix}", 0, 179, (0, 179))
        k = st.slider(f"Kernel (Noise Filter) - {prefix}", 1, 20, 1)
    with col2:
        s_range = st.slider(f"Saturation (S) - {prefix}", 0, 255, (0, 255))
        inverter = st.checkbox(f"Invert Mask - {prefix}")
    with col3:
        v_range = st.slider(f"Value (V) - {prefix}", 0, 255, (0, 255))
        
    lower = np.array([h_range[0], s_range[0], v_range[0]])
    upper = np.array([h_range[1], s_range[1], v_range[1]])
    
    mask = cv2.inRange(hsv, lower, upper)
    if inverter:
        mask = cv2.bitwise_not(mask)
        
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    area_total = mask.shape[0] * mask.shape[1]
    pixels_cobertos = np.sum(mask == 255)
    cobertura = (pixels_cobertos / area_total) * 100
    
    st.markdown(f"**Coverage:** {cobertura:.2f}% ({pixels_cobertos} voxels)")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(result, caption=f"{prefix} Isolated", use_container_width=True)
    with col_img2:
        st.image(mask, caption=f"{prefix} Binary Mask", use_container_width=True)
        
    mask_bin = (mask > 127).astype(np.uint8)
    return mask_bin

if geometry_type == "IRVM (Image Reconstructed Voxel Model)":
    st.sidebar.markdown("**1. Upload Microscopy Image**")
    uploaded_image = st.sidebar.file_uploader("Upload Image (.png, .jpg, .tif)", type=['png', 'jpg', 'jpeg', 'tif'])
    downsample = st.sidebar.number_input("Downsample Factor", min_value=1, max_value=20, value=1)
    
    if uploaded_image:
        def load_rgb_image(uploaded_file, ds):
            img = Image.open(uploaded_file).convert('RGB')
            arr = np.array(img)
            if ds > 1: arr = arr[::ds, ::ds]
            return arr
            
        img_rgb = load_rgb_image(uploaded_image, downsample)
        
        st.markdown("## Step 1: Monolayer (Base) Segmentation")
        st.info("Adjust the sliders to isolate the Monolayer.")
        mask_base = st_hsv_segmentation(img_rgb, "Monolayer")
        st.markdown("---")
        st.markdown("## Step 2: Islands (Stack) Segmentation")
        st.info("Now, adjust the sliders to isolate ONLY the thicker islands/multilayers.")
        mask_stack = st_hsv_segmentation(img_rgb, "Islands")
        
        st.sidebar.markdown("**2. 3D Construction Parameters**")
        nz_base = st.sidebar.number_input("Base Thickness (Z)", min_value=1, value=1)
        nz_stack = st.sidebar.number_input("Stack Thickness (Z)", min_value=1, value=5)
        ri_base = st.sidebar.number_input("Base Index (ri)", min_value=1, value=1)
        ri_stack = st.sidebar.number_input("Stack Index (ri)", min_value=1, value=2)
        
        if st.sidebar.button("Generate IRVM Matrix", type="primary"):
            if mask_base.shape != mask_stack.shape:
                st.sidebar.error(f"Error: Dimensions mismatch.")
            else:
                # Salva na memória de sessão
                st.session_state.matriz_obj = IRVMMatrix(mask_base, mask_stack, nz_base, nz_stack, ri_base, ri_stack)

elif geometry_type == "Simple Cube (with vacuum padding)":
    st.sidebar.markdown("**1. Box Dimensions (Total Matrix)**")
    col1, col2, col3 = st.sidebar.columns(3)
    nx = col1.number_input("Matrix X", value=50)
    ny = col2.number_input("Matrix Y", value=50)
    nz = col3.number_input("Matrix Z", value=50)
    
    st.sidebar.markdown("**2. Material Dimensions (Cube)**")
    col4, col5, col6 = st.sidebar.columns(3)
    cx = col4.number_input("Cube X", value=20)
    cy = col5.number_input("Cube Y", value=20)
    cz = col6.number_input("Cube Z", value=10)
    
    z_offset = st.sidebar.number_input("Z-Offset", value=0)
    ri = st.sidebar.number_input("Material Index (ri)", value=1)
    
    if st.sidebar.button("Generate Matrix"):
        if cx > nx or cy > ny or (cz + z_offset) > nz:
            st.sidebar.error("Error: The cube is out of bounds.")
        else:
            st.session_state.matriz_obj = Cubo(nx, ny, nz, cx, cy, cz, z_offset, ri)

elif geometry_type == "Porous 2D Membrane":
    st.sidebar.markdown("**1. Matrix & Membrane Dimensions**")
    col1, col2, col3 = st.sidebar.columns(3)
    nx = col1.number_input("Matrix X", value=1000)
    ny = col2.number_input("Matrix Y", value=1000)
    nz = col3.number_input("Matrix Z", value=20)
    
    col4, col5 = st.sidebar.columns(2)
    esp = col4.number_input("Membrane Thickness (Z)", min_value=1, value=7)
    ri = col5.number_input("Material Index", min_value=1, value=1)
    
    if "pores_config" not in st.session_state:
        st.session_state.pores_config = []
        
    st.sidebar.markdown("**2. Add Nanopores (Spherical)**")
    col_p1, col_p2 = st.sidebar.columns(2)
    p_radius = col_p1.number_input("Radius (R)", min_value=1, value=25)
    p_qty = col_p2.number_input("Quantity", min_value=1, value=10)
    
    col_b1, col_b2 = st.sidebar.columns(2)
    if col_b1.button("➕ Add Pore Type"):
        st.session_state.pores_config.append((p_radius, p_qty))
    if col_b2.button("🗑️ Clear Pores"):
        st.session_state.pores_config = []
        
    st.sidebar.markdown("**Current Pores Overview:**")
    if not st.session_state.pores_config:
        st.sidebar.text("No pores added yet.")
    for i, (r, q) in enumerate(st.session_state.pores_config):
        st.sidebar.text(f"Type {i+1}: Radius={r} | Quantity={q}")
        
    st.sidebar.markdown("**3. Generation Settings**")
    seed = st.sidebar.number_input("Random Seed (Reproducibility)", value=42)
    
    if st.sidebar.button("Generate Porous Matrix", type="primary"):
        st.session_state.matriz_obj = MembranaPorosa(nx, ny, nz, esp, ri, st.session_state.pores_config, seed)

elif geometry_type == "Advanced Stack Builder":
    st.sidebar.markdown("**1. Box Limits (Total Matrix Area)**")
    col1, col2 = st.sidebar.columns(2)
    nx = col1.number_input("Max Matrix X", value=60)
    ny = col2.number_input("Max Matrix Y", value=60)
    
    if "adv_stack_layers" not in st.session_state:
        st.session_state.adv_stack_layers = []
        
    st.sidebar.markdown("**2. Add Custom Layer (Bottom to Top)**")
    
    # Tamanho da Camada
    col_size1, col_size2, col_size3 = st.sidebar.columns(3)
    layer_cx = col_size1.number_input("Width (X)", min_value=1, value=60, key="adv_x")
    layer_cy = col_size2.number_input("Depth (Y)", min_value=1, value=60, key="adv_y")
    layer_cz = col_size3.number_input("Thickness (Z)", min_value=1, value=10, key="adv_z")
    
    # Material e Alinhamento
    col_al1, col_al2 = st.sidebar.columns(2)
    layer_ri = col_al1.number_input("Material Index", min_value=0, value=1, key="adv_ri")
    layer_align = col_al2.selectbox("Alignment", ["Center", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Custom"], key="adv_al")
    
    # Se escolher Custom, exibe caixas para digitar as coordenadas manuais
    offset_x, offset_y = 0, 0
    if layer_align == "Custom":
        col_off1, col_off2 = st.sidebar.columns(2)
        offset_x = col_off1.number_input("Offset X", value=0)
        offset_y = col_off2.number_input("Offset Y", value=0)
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("➕ Add Layer"):
        st.session_state.adv_stack_layers.append({
            'cx': layer_cx, 'cy': layer_cy, 'cz': layer_cz, 
            'ri': layer_ri, 'align': layer_align, 
            'offset_x': offset_x, 'offset_y': offset_y
        })
    
    if col_btn2.button("🗑️ Clear Stack"):
        st.session_state.adv_stack_layers = []
        
    st.sidebar.markdown("**Current Advanced Stack:**")
    total_z = 0
    for i, layer in enumerate(st.session_state.adv_stack_layers):
        al = layer['align']
        st.sidebar.text(f"L{i+1}: {layer['cx']}x{layer['cy']}x{layer['cz']} | {al} | ri={layer['ri']}")
        total_z += layer['cz']
    st.sidebar.info(f"Total Box Size: {nx}x{ny}x{total_z}")
        
    if st.sidebar.button("Generate Matrix", type="primary"):
        if not st.session_state.adv_stack_layers:
            st.sidebar.error("Error: Add at least one layer first!")
        else:
            st.session_state.matriz_obj = AdvancedStackMatrix(nx, ny, st.session_state.adv_stack_layers)

elif geometry_type == "Cube with Top Film":
    st.sidebar.markdown("**1. Box Dimensions (Total Matrix)**")
    col1, col2, col3 = st.sidebar.columns(3)
    nx = col1.number_input("Matrix X", value=50)
    ny = col2.number_input("Matrix Y", value=50)
    nz = col3.number_input("Matrix Z", value=50)
    
    st.sidebar.markdown("**2. Width/Depth (X, Y)**")
    col4, col5 = st.sidebar.columns(2)
    cx = col4.number_input("Structure X", value=20)
    cy = col5.number_input("Structure Y", value=20)
    
    st.sidebar.markdown("**3. Layers (Z) & Materials**")
    col_base1, col_base2 = st.sidebar.columns(2)
    cz_base = col_base1.number_input("Base Thickness", value=15)
    ri_base = col_base2.number_input("Base Index", value=1)
    
    col_film1, col_film2 = st.sidebar.columns(2)
    cz_film = col_film1.number_input("Film Thickness", value=5)
    ri_film = col_film2.number_input("Film Index", value=2)
    
    z_offset = st.sidebar.number_input("Z-Offset", value=0)
    
    if st.sidebar.button("Generate Matrix", type="primary"):
        if cx > nx or cy > ny or (cz_base + cz_film + z_offset) > nz:
            st.sidebar.error("Error: The structure exceeds the matrix dimensions.")
        else:
            st.session_state.matriz_obj = CuboMulticamada(nx, ny, nz, cx, cy, cz_base, cz_film, z_offset, ri_base, ri_film)

elif geometry_type == "Triangular Prism":
    base = st.sidebar.number_input("Base (points)", min_value=1, max_value=200, value=30)
    altura = st.sidebar.number_input("Height (points)", min_value=1, max_value=200, value=50)
    ri = st.sidebar.number_input("Material Index (ri)", min_value=1, value=1)
    if st.sidebar.button("Generate Matrix"):
        st.session_state.matriz_obj = PrismaTriangular(base, altura, ri)

elif geometry_type == "Core/Shell Cube":
    lado_total = st.sidebar.number_input("Total Side", value=40)
    lado_core = st.sidebar.number_input("Core Side", value=20)
    altura = st.sidebar.number_input("Height (Z)", value=40)
    col1, col2 = st.sidebar.columns(2)
    ri1 = col1.number_input("Core Index", value=2)
    ri2 = col2.number_input("Shell Index", value=1)
    if st.sidebar.button("Generate Matrix"):
        st.session_state.matriz_obj = CuboCoreShell(lado_total, lado_core, altura, ri1, ri2)

elif geometry_type == "Buried Cylinder":
    x = st.sidebar.number_input("X Dimension", value=40)
    y = st.sidebar.number_input("Y Dimension", value=40)
    h_cilindro = st.sidebar.number_input("Cylinder Height (H)", value=20)
    h_total = st.sidebar.number_input("Total Height (h)", value=40)
    raio = st.sidebar.number_input("Radius (R)", value=10.0)
    col1, col2 = st.sidebar.columns(2)
    ri = col1.number_input("Cylinder Index", value=1)
    hi = col2.number_input("Substrate Index", value=2)
    if st.sidebar.button("Generate Matrix"):
        st.session_state.matriz_obj = CilindroEnterrado(x, y, h_cilindro, h_total, raio, ri, hi)

elif geometry_type == "Core/Shell Sphere":
    r_shell = st.sidebar.number_input("Shell Radius (Total)", value=29.0)
    r_core = st.sidebar.number_input("Core Radius", value=25.0)
    col1, col2 = st.sidebar.columns(2)
    i2 = col1.number_input("Core Index", value=2)
    i1 = col2.number_input("Shell Index", value=1)
    if st.sidebar.button("Generate Matrix"):
        st.session_state.matriz_obj = EsferaCoreShell(r_shell, r_core, i1, i2)

elif geometry_type == "Spheroid on Substrate":
    a = st.sidebar.number_input("X Curvature (a)", value=20)
    b = st.sidebar.number_input("Y Curvature (b)", value=20)
    c = st.sidebar.number_input("Z Curvature (c)", value=30)
    h_subs = st.sidebar.number_input("Substrate Height (h)", value=15)
    half = st.sidebar.checkbox("Only upper half (-half)?")
    col1, col2 = st.sidebar.columns(2)
    ri = col1.number_input("Spheroid Index", value=1)
    hi = col2.number_input("Substrate Index", value=2)
    if st.sidebar.button("Generate Matrix"):
        st.session_state.matriz_obj = EsferoideSubstrato(a, b, c, h_subs, ri, hi, half)


elif geometry_type == "Universal Stack Builder":
    st.sidebar.markdown("**1. Set Base Dimensions**")
    col1, col2 = st.sidebar.columns(2)
    nx = col1.number_input("Matrix X", value=50)
    ny = col2.number_input("Matrix Y", value=50)
    
    # Cria uma lista vazia na memória se ela não existir
    if "stack_layers" not in st.session_state:
        st.session_state.stack_layers = []
        
    st.sidebar.markdown("**2. Add Layers (Bottom to Top)**")
    col3, col4 = st.sidebar.columns(2)
    layer_z = col3.number_input("Thickness (Z)", min_value=1, value=10)
    layer_ri = col4.number_input("Material Index (ri)", min_value=0, value=1)
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("➕ Add Layer"):
        st.session_state.stack_layers.append({'z': layer_z, 'ri': layer_ri})
    
    if col_btn2.button("🗑️ Clear Stack"):
        st.session_state.stack_layers = []
        
    st.sidebar.markdown("**Current Stack Overview:**")
    total_z = 0
    for i, layer in enumerate(st.session_state.stack_layers):
        st.sidebar.text(f"Layer {i+1}: Thickness={layer['z']}, Index={layer['ri']}")
        total_z += layer['z']
    st.sidebar.info(f"Total Matrix Z will be: {total_z}")
        
    if st.sidebar.button("Generate Stack Matrix", type="primary"):
        if not st.session_state.stack_layers:
            st.sidebar.error("Error: Add at least one layer first!")
        else:
            st.session_state.matriz_obj = StackBuilderMatrix(nx, ny, st.session_state.stack_layers)

elif geometry_type == "Import & View (.mtx)":
    st.sidebar.markdown("**Upload existing PowerMEIS matrix**")
    uploaded_mtx = st.sidebar.file_uploader("Upload .mtx", type=['mtx', 'txt'])
    
    if uploaded_mtx:
        if st.sidebar.button("Load & View Matrix", type="primary"):
            # Decodifica o arquivo de texto
            content = uploaded_mtx.read().decode("utf-8").splitlines()
            header = content[0].strip().split()
            nx, ny, nz = map(int, header[:3])
            
            # Lógica RLE (Run-Length Encoding) do PowerMEIS
            data = []
            for line in content[1:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    n, i = map(int, parts)
                    data.extend([i] * n)
            
            arr = np.array(data, dtype=np.uint8)
            expected = nx * ny * nz
            
            # Tratamento de segurança caso o arquivo venha com tamanho incorreto
            if len(arr) > expected: arr = arr[:expected]
            elif len(arr) < expected: arr = np.pad(arr, (0, expected - len(arr)), 'constant')
            
            # Remonta a matriz 3D
            arr = arr.reshape((nz, ny, nx))
            
            # Como a exportação do PowerMEIS escreve do topo (nz-1) para o fundo (0),
            # nós invertemos o eixo Z na hora de ler para o visualizador ficar correto
            arr = np.flip(arr, axis=0) 
            
            st.session_state.matriz_obj = ImportedMatrix(arr)
            st.toast("Matrix loaded successfully!", icon="✅")

# ==========================================
# MAIN AREA: VISUALIZATION & EXPORT
# ==========================================
if st.session_state.matriz_obj is not None:
    # Resgata a matriz da memória
    matriz_obj_atual = st.session_state.matriz_obj
    
    st.markdown("---")
    st.subheader(f"3D Visualization")
    
    matriz = matriz_obj_atual.matriz
    x_coords, y_coords, z_coords = np.where(matriz > 0)
    valores = matriz[matriz > 0]
    
    total_points = len(x_coords)
    if total_points > 100000:
        st.warning(f"Large matrix detected ({total_points} voxels). Downsampling 3D plot for browser performance. **The exported file will remain in full resolution.**")
        step = total_points // 50000 
        x_coords, y_coords, z_coords, valores = x_coords[::step], y_coords[::step], z_coords[::step], valores[::step]
    
    fig = go.Figure(data=go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=3,
            color=valores,                
            colorscale='Viridis',
            opacity=0.8,
            symbol='square'               
        )
    ))
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_aspectmode='data')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Export Matrix")
    
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        default_name = geometry_type.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').lower()
        # Quando você digita aqui, o app recarrega, mas o st.session_state mantém a matriz viva!
        custom_name = st.text_input("Filename (without .mtx):", value=default_name)
    
    filename = f"{custom_name}.mtx"
    
    # Gera/sobrescreve o arquivo no sistema toda vez que o nome é alterado
    matriz_obj_atual.export_powermeis(filename, compactado=True)
    
    with open(filename, "rb") as file:
        st.download_button(
            label=f"Download {filename}",
            data=file,
            file_name=filename,
            mime="text/plain"
        )