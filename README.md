# 🚀 Sistema de Reconhecimento Facial Inteligente com PyTorch 🧠

---

## ✨ Visão Geral do Projeto

Este projeto tem como objetivo principal desenvolver um sistema robusto de reconhecimento facial utilizando as capacidades avançadas do PyTorch e Redes Neurais Convolucionais (CNNs). Além do reconhecimento padrão, o sistema incorpora uma funcionalidade inovadora de **cadastro incremental de usuários**, permitindo a inclusão de novas identidades sem a necessidade de retreinar o modelo do zero.

### 🎯 Objetivo Principal (do Professor)

Desenvolver um sistema de reconhecimento facial utilizando PyTorch e redes neurais convolucionais (CNN). O foco do projeto é aplicar CNNs, com ou sem transfer learning, para identificação de pessoas a partir de imagens de rosto.

### 📜 Regras e Restrições Técnicas (do Professor)

- O projeto deve ser implementado com PyTorch.
- A arquitetura deve obrigatoriamente utilizar redes convolucionais (CNN).
- É permitido o uso de modelos pré-treinados como ResNet, VGGFace, FaceNet, EfficientNet, etc.
- São aceitas tanto abordagens de classificação direta (com imagens de rostos recortados) quanto de detecção de objetos, como YOLO.
- Pode-se usar bibliotecas como OpenCV, MTCNN, MediaPipe, ou YOLO para detecção facial.
- A entrada do modelo pode ser a imagem completa ou o recorte facial, conforme a abordagem adotada.

### ✅ Requisitos Funcionais (do Professor)

- O sistema deve reconhecer corretamente cada participante com base em uma imagem contendo seu rosto.
- A entrada pode ser uma imagem com um ou mais rostos.
- A saída deve indicar o nome ou rótulo da pessoa reconhecida.

---

## 🛠️ Detalhes da Implementação

Aqui, detalhamos o coração técnico do nosso sistema.

### 🏗️ Arquitetura da Rede Neural

Nosso sistema utiliza uma **Rede Neural Convolucional (CNN)** baseada na arquitetura pré-treinada **ResNet18**. A ResNet18 é conhecida por sua eficiência e capacidade de aprender representações hierárquicas complexas de imagens.

- **Backbone**: ResNet18 pré-treinada em ImageNet. As camadas iniciais (backbone) são congeladas durante o fine-tuning para preservar o conhecimento de características gerais.
- **Embedding Head**: Uma camada convolucional e/ou linear adicional (`nn.Sequential` em `ImprovedFaceRecognitionModel`) que mapeia as features extraídas pelo backbone para um vetor de embedding de tamanho `EMBEDDING_SIZE` (512 no nosso caso). Este embedding serve como uma representação numérica única do rosto.
- **Classificador**: Uma camada linear final que mapeia o vetor de embedding para o número de classes (pessoas) presentes no dataset. Esta camada é adaptada dinamicamente quando novos usuários são cadastrados.

### ⚙️ Estratégia de Pré-processamento e Treinamento

Nosso pipeline de treinamento é cuidadosamente projetado para otimizar o aprendizado e a adaptabilidade do modelo.

#### Pré-processamento de Imagens

As imagens passam por um pipeline de pré-processamento robusto:
1.  **Redimensionamento**: Todas as imagens são redimensionadas para `224x224` pixels, o tamanho de entrada esperado pela ResNet18.
2.  **Conversão para Tensor**: Imagens são convertidas para tensores PyTorch.
3.  **Normalização**: Aplicamos a normalização padrão da ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) para alinhar a distribuição dos pixels com os dados em que o modelo foi pré-treinado.
4.  **Carregamento de Imagens Seguras**: Para lidar com caminhos de arquivo que contenham caracteres especiais (comuns no Windows), utilizamos a biblioteca `Pillow` (`PIL.Image.open`) para carregar as imagens, que é mais robusta que `cv2.imread` para esses casos.

#### Treinamento do Modelo

-   **Otimizador**: `AdamW` é utilizado como otimizador, conhecido por sua boa performance e tratamento de decaimento de peso.
-   **Taxa de Aprendizado (Learning Rate)**: `LEARNING_RATE = 0.0001` para o treinamento inicial.
-   **Agendador de Taxa de Aprendizado (Scheduler)**: `ReduceLROnPlateau` monitora a acurácia de validação. Se a acurácia não melhorar por um determinado número de épocas (`patience`), a taxa de aprendizado é reduzida, ajudando o modelo a convergir.
-   **Critério de Perda (Loss Function)**: `nn.CrossEntropyLoss` é usada para classificação multi-classe.
-   **Épocas**: Definido como `EPOCHS = 1` para testes rápidos. Para um modelo robusto, um número maior de épocas (e.g., 20-50) seria necessário.
-   **Early Stopping**: Um mecanismo de early stopping com `patience = 15` monitora a acurácia de validação para evitar overfitting, parando o treinamento se a performance não melhorar.

### 📈 Resultados do Treinamento

Após o treinamento, o sistema salva o modelo com a melhor acurácia de validação. Os resultados incluem:

-   **Acurácia de Treinamento e Validação**: Monitoradas ao longo das épocas.
-   **Perda de Treinamento**: Monitorada para verificar a convergência.
-   **Matriz de Confusão**: Uma matriz de confusão é gerada e salva (ou exibida) para visualizar o desempenho do classificador em cada classe, identificando erros e acertos.

---

## ➕ Adicional: MVP para Cadastro Incremental de Usuários

Este é um diferencial crucial do nosso projeto, permitindo a **evolução contínua** do sistema.

### 🎯 Objetivo (do Professor)

Permitir a inclusão de novos usuários no sistema de reconhecimento facial sem apagar ou sobrescrever os usuários já cadastrados.

### 📋 Requisitos Técnicos (do Professor)

- O sistema deve armazenar as representações faciais dos usuários (por exemplo, vetores de embeddings ou imagens associadas aos rótulos).
    - **Nossa Abordagem**: Armazenamos as imagens diretamente em diretórios nomeados e o modelo aprende os embeddings internamente.
- O cadastro pode ser feito a partir de uma ou mais imagens do novo usuário.
    - **Nossa Abordagem**: A interface permite upload de múltiplas imagens.
- O modelo pode:
    - Atualizar um banco de embeddings, se a arquitetura utilizar essa abordagem.
    - Utilizar fine-tuning leve ou incremental em uma rede existente, se necessário.
        - **Nossa Abordagem**: Utilizamos fine-tuning leve.
    - Atualizar um banco de dados com imagens e rótulos, e reprocessar conforme necessário.
        - **Nossa Abordagem**: As imagens são copiadas para as pastas de `Fotos/` e o mapeamento de labels (`label_mapping_fixed.json`) é atualizado.

### 🚀 Funcionalidades Implementadas (no MVP)

-   **Interface Gráfica Simples (Streamlit)**: Uma seção dedicada no `app.py` permite que o usuário insira o nome da nova pessoa e faça upload de suas imagens.
-   **Associação Nome/Rótulo**: As imagens são salvas em uma pasta com o nome do usuário, e o sistema automaticamente associa este nome a uma nova classe/rótulo.
-   **Fine-tuning Leve**:
    1.  O modelo pré-treinado é carregado.
    2.  Se houver uma nova classe (novo usuário), a camada classificadora final do modelo é adaptada para incluir essa nova classe.
    3.  As camadas do **backbone (ResNet18) são congeladas**. Isso impede que os pesos das características gerais sejam alterados.
    4.  Apenas o **embedding head e a nova camada classificadora são treinados** (descongelados) com uma taxa de aprendizado menor. Isso acelera o treinamento e foca na adaptação às novas identidades.
-   **Garantia de Reconhecimento Anterior**: A estratégia de fine-tuning leve minimiza o impacto no reconhecimento de usuários já cadastrados, pois o conhecimento pré-existente do backbone é preservado. Testes práticos após a inclusão de novos usuários são recomendados para validar isso.
-   **Registro Persistente**: O modelo treinado (`face_recognition_model_fixed.pt`) e o mapeamento de labels (`label_mapping_fixed.json`) são salvos e atualizados após cada processo de fine-tuning, garantindo que as informações dos novos usuários sejam mantidas.

---

## ⚙️ Configuração do Ambiente

1.  **Clone este repositório:**
    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd [NOME_DO_DIRETORIO]
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Para o treinamento inicial do modelo:**
    (Este passo é necessário uma vez para gerar o modelo base antes de usar a interface Streamlit)
    ```bash
    python rooney_trabalho.py
    ```

---

## 🚀 Como Usar a Aplicação

1.  **Inicie a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```
    A interface web será aberta automaticamente no seu navegador padrão.

2.  **Funcionalidades na Interface:**
    -   **Reconhecimento Facial**: Faça upload de uma imagem contendo um rosto para que o sistema tente identificá-lo.
    -   **Cadastro Incremental de Novos Usuários**:
        1.  Na seção "Cadastro Incremental de Novos Usuários", digite o **Nome do Novo Usuário**.
        2.  Faça upload de **pelo menos 5 imagens** do novo usuário (quanto mais imagens de boa qualidade, melhor a performance).
        3.  Clique em "**Cadastrar Novo Usuário e Fine-tune**".
        4.  Aguarde o processo de fine-tuning (pode levar alguns minutos).
        5.  **Importante**: Após o cadastro, **recarregue a página do Streamlit** no seu navegador para que o novo modelo (com as novas classes) seja carregado e utilizado para reconhecimento.

---

## 💡 Dicas para Melhor Performance

-   **Qualidade das Imagens**: Use imagens de boa qualidade para o cadastro e reconhecimento.
-   **Número de Imagens**: Forneça pelo menos 5-10 imagens distintas por usuário para o cadastro (quanto mais, melhor).
-   **Iluminação e Posição**: Certifique-se de que as imagens tenham boa iluminação e que o rosto esteja bem visível.
-   **Recorte Facial**: Embora o sistema lide com redimensionamento, imagens com rostos bem enquadrados e centralizados tendem a performar melhor.

---

## ⁉️ Solução de Problemas Comuns

1.  **`ModuleNotFoundError`**: Certifique-se de que todas as dependências em `requirements.txt` estão instaladas e que o ambiente virtual está ativado. Renomeie `Rooney trabalho.py` para `rooney_trabalho.py` se ainda não o fez.
2.  **`expected scalar type Double but found Float`**: Execute `python rooney_trabalho.py` para retreinar o modelo com a precisão correta antes de rodar `streamlit run app.py`.
3.  **Erro ao Carregar Imagens (`cv2.imread` falha com caracteres especiais)**: A solução com `Pillow` foi implementada. Verifique se as imagens não estão corrompidas.
4.  **Erro de CUDA**: Se você não tiver uma GPU NVIDIA, o sistema funcionará em CPU, mas será mais lento. Certifique-se de que as bibliotecas `torch` e `torchvision` foram instaladas para CPU se não tiver CUDA.
5.  **Webcam não funciona**: Certifique-se de que sua webcam está funcionando e acessível por outros aplicativos. Permissões do navegador podem ser necessárias. (Se você adicionar a funcionalidade de webcam).
6.  **Erros de Memória**: Se encontrar erros de memória, tente reduzir o `BATCH_SIZE` em `rooney_trabalho.py` ou o número de `num_workers` nos `DataLoader`s.

---

## 📂 Estrutura de Diretórios

```
projeto/
├── app.py                     # 🚀 Interface web principal (Streamlit)
├── rooney_trabalho.py         # 🧠 Core do sistema: modelo, treinamento e fine-tuning
├── requirements.txt           # 📦 Dependências do Python
├── .gitignore                 # 🚫 Arquivos e diretórios a serem ignorados pelo Git
├── README.md                  # 📄 Este arquivo: documentação do projeto
└── data/                      # 🗃️ Diretório para dados do projeto
    ├── raw/                   # 📸 Imagens originais dos usuários (mantém subpastas por pessoa)
    │   └── .gitkeep
    ├── processed/             # ✨ Imagens pré-processadas (geradas, se aplicável)
    │   └── .gitkeep
    └── models/                # 💾 Modelos PyTorch treinados e mapeamentos
        └── .gitkeep
```

---

## 🤝 Contribuindo

Sinta-se à vontade para abrir [issues]([URL_DO_REPOSITORIO]/issues) para relatar bugs ou sugerir melhorias, ou enviar [pull requests]([URL_DO_REPOSITORIO]/pulls) com novas funcionalidades.

---

## 📄 Licença

Este projeto está sob a licença [MIT License](https://opensource.org/licenses/MIT). 