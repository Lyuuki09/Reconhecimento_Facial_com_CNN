import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import json
from rooney_trabalho import ImprovedFaceRecognitionModel, FaceDataset, DEVICE

# Configuração da página
st.set_page_config(
    page_title="Sistema de Reconhecimento Facial",
    page_icon="👤",
    layout="wide"
)

# Título e descrição
st.title("Sistema de Reconhecimento Facial com CNN")
st.markdown("""
Este sistema utiliza redes neurais convolucionais (CNN) para reconhecimento facial.
Carregue uma imagem contendo um rosto para identificar a pessoa.
""")

# Carregar o modelo e mapeamento de labels
@st.cache_resource
def load_model():
    try:
        # Carregar mapeamento de labels
        with open("label_mapping_fixed.json", "r") as f:
            label_mapping = json.load(f)
        
        # Inicializar modelo
        model = ImprovedFaceRecognitionModel(num_classes=len(label_mapping['label_to_idx']))
        model.load_state_dict(torch.load("face_recognition_model_fixed.pt", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # Garante que o modelo esteja em float32, que é o tipo padrão para a maioria das redes neurais
        model.float()
        
        return model, label_mapping
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None, None

# Função para pré-processar a imagem
def preprocess_image(image):
    # Converter para RGB se necessário
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Redimensionar
    image = cv2.resize(image, (224, 224))
    
    # Normalizar
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Converter para tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image

# Interface principal
def main():
    # Carregar modelo
    model, label_mapping = load_model()
    
    if model is None:
        st.error("Por favor, treine o modelo primeiro executando o script principal.")
        return
    
    # Criar duas colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload de Imagem")
        uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Ler e exibir a imagem
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_column_width=True)
            
            # Converter para array numpy
            image_np = np.array(image)
            
            # Botão para reconhecimento
            if st.button("Reconhecer Rosto"):
                with st.spinner("Processando..."):
                    try:
                        # Pré-processar imagem
                        processed_image = preprocess_image(image_np)
                        processed_image = processed_image.to(DEVICE)
                        
                        # Fazer predição
                        with torch.no_grad():
                            logits, _ = model(processed_image)
                            probabilities = torch.softmax(logits, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                        
                        # Obter resultado
                        predicted_label = label_mapping['idx_to_label'][str(predicted.item())]
                        confidence_value = confidence.item() * 100
                        
                        # Exibir resultado
                        with col2:
                            st.subheader("Resultado do Reconhecimento")
                            st.success(f"Pessoa identificada: {predicted_label}")
                            st.info(f"Confiança: {confidence_value:.2f}%")
                            
                            # Exibir barra de progresso para confiança
                            st.progress(confidence_value / 100)
                            
                            # Exibir todas as probabilidades
                            st.subheader("Probabilidades para cada pessoa:")
                            probs = probabilities[0].cpu().numpy()
                            for idx, prob in enumerate(probs):
                                label = label_mapping['idx_to_label'][str(idx)]
                                st.write(f"{label}: {prob*100:.2f}%")
                    
                    except Exception as e:
                        st.error(f"Erro durante o reconhecimento: {str(e)}")
    
    # Seção de informações
    st.markdown("---")
    st.subheader("Informações do Sistema")
    st.markdown("""
    - Modelo: ResNet18 com fine-tuning
    - Tamanho da imagem: 224x224 pixels
    - Classes: {}
    """.format(len(label_mapping['label_to_idx'])))

    st.markdown("--- ")
    st.subheader("Cadastro Incremental de Novos Usuários")
    st.markdown("""
    Cadastre novas pessoas no sistema de reconhecimento facial sem retreinar o modelo do zero. 
    Após o cadastro, o modelo será fine-tunado automaticamente para incluir o novo usuário.
    """)

    new_user_name = st.text_input("Nome do Novo Usuário")
    new_user_images = st.file_uploader("Faça upload de imagens do novo usuário (mínimo 5 imagens para boa performance)", 
                                        type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Cadastrar Novo Usuário e Fine-tune"):
        if not new_user_name:
            st.error("Por favor, digite o nome do novo usuário.")
        elif not new_user_images:
            st.error("Por favor, faça upload de pelo menos uma imagem para o novo usuário.")
        else:
            # Salvar as imagens temporariamente para que o rooney_trabalho possa acessá-las
            temp_dir = "temp_new_user_images"
            os.makedirs(temp_dir, exist_ok=True)
            uploaded_file_paths = []
            for uploaded_file in new_user_images:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_file_paths.append(temp_file_path)
            
            try:
                with st.spinner(f"Cadastrando {new_user_name} e fine-tunando o modelo... Isso pode levar alguns minutos."):
                    # Chamar a função de fine-tuning do seu script principal
                    from rooney_trabalho import add_new_user_and_finetune
                    add_new_user_and_finetune(new_user_name, uploaded_file_paths)
                st.success(f"Usuário '{new_user_name}' cadastrado e modelo fine-tunado com sucesso!")
                st.info("Recarregue a página para que o novo modelo seja utilizado para reconhecimento.")
            except Exception as e:
                st.error(f"Erro ao cadastrar novo usuário: {e}")
            finally:
                # Limpar arquivos temporários
                for f_path in uploaded_file_paths:
                    os.remove(f_path)
                os.rmdir(temp_dir) # Remove o diretório se estiver vazio

if __name__ == "__main__":
    main() 