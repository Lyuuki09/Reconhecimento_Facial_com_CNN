# Sistema de Reconhecimento Facial com CNN

Este projeto implementa um sistema de reconhecimento facial utilizando Redes Neurais Convolucionais (CNN) com PyTorch. O sistema permite reconhecer pessoas a partir de imagens de rosto e suporta o cadastro incremental de novos usuários.

## 🚀 Funcionalidades

- Reconhecimento facial em tempo real
- Interface web amigável usando Streamlit
- Cadastro incremental de novos usuários
- Fine-tuning automático do modelo
- Visualização de probabilidades de reconhecimento
- Suporte a múltiplas imagens por usuário

## 📋 Pré-requisitos

- Python 3.8 ou superior
- CUDA (opcional, para aceleração GPU)
- Webcam (opcional, para captura em tempo real)

## ⚙️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Lyuuki09/Reconhecimento_Facial_com_CNN.git
cd Reconhecimento_Facial_com_CNN
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📁 Estrutura de Diretórios

```
Reconhecimento_Facial_com_CNN/
├── Fotos/                    # Diretório para armazenar fotos dos usuários
│   └── [nome_pessoa]/       # Subdiretório para cada pessoa
├── app.py                   # Interface web Streamlit
├── rooney_trabalho.py       # Core do sistema e treinamento
├── requirements.txt         # Dependências do projeto
└── README.md               # Este arquivo
```

## 🎯 Como Usar

### 1. Preparação Inicial

1. Crie uma pasta dentro do diretório `Fotos/` para cada pessoa que deseja cadastrar
2. Adicione as fotos da pessoa na respectiva pasta
   - Recomendado: 40-50 fotos por pessoa
   - Formato: JPG, JPEG ou PNG
   - Tamanho recomendado: mínimo 224x224 pixels

### 2. Treinamento do Modelo

Execute o script principal para treinar o modelo:
```bash
python rooney_trabalho.py
```

### 3. Iniciando a Interface Web

Execute o aplicativo Streamlit:
```bash
streamlit run app.py
```

A interface web será aberta automaticamente no seu navegador padrão.

### 4. Reconhecimento Facial

1. Na interface web, clique em "Escolha uma imagem..."
2. Selecione uma imagem contendo um rosto
3. Clique em "Reconhecer Rosto"
4. O sistema mostrará:
   - A pessoa identificada
   - Nível de confiança
   - Probabilidades para todas as pessoas cadastradas

### 5. Cadastro de Novos Usuários

1. Na seção "Cadastro Incremental de Novos Usuários":
   - Digite o nome do novo usuário
   - Faça upload de pelo menos 5 imagens do usuário (ele funciona bem, mas recomendado são 50 imagens)
   - Clique em "Cadastrar Novo Usuário e Fine-tune"
2. Aguarde o processo de fine-tuning
3. Recarregue a página e o arquivo de treinamento para que o novo modelo seja carregado

## 💡 Dicas para Melhor Performance

- Use fotos de boa qualidade e bem iluminadas
- Certifique-se que o rosto está bem visível e centralizado
- Evite fotos com múltiplos rostos
- Use diferentes ângulos e expressões para melhor treinamento
- Mantenha um fundo neutro nas fotos

## ⚠️ Solução de Problemas

1. **Erro ao carregar o modelo**:
   - Verifique se o arquivo `face_recognition_model_fixed.pt` existe
   - Execute `rooney_trabalho.py` para gerar o modelo

2. **Erro de CUDA**:
   - O sistema funcionará em CPU se CUDA não estiver disponível
   - Verifique se as bibliotecas PyTorch estão instaladas corretamente

3. **Erro ao carregar imagens**:
   - Verifique se as imagens não estão corrompidas
   - Use formatos suportados (JPG, JPEG, PNG)

## 📝 Detalhes Técnicos

### Arquitetura da Rede Neural

- **Backbone**: ResNet18 pré-treinada em ImageNet
- **Embedding Head**: Camada adicional que mapeia features para vetor de embedding (512 dimensões)
- **Classificador**: Camada linear final adaptável para número de classes

### Pipeline de Treinamento

1. **Pré-processamento**:
   - Redimensionamento para 224x224 pixels
   - Normalização com valores ImageNet
   - Conversão para tensores PyTorch

2. **Treinamento**:
   - Otimizador: AdamW
   - Learning Rate: 0.0001
   - Scheduler: ReduceLROnPlateau
   - Loss Function: CrossEntropyLoss
   - Early Stopping com patience = 15

3. **Fine-tuning para Novos Usuários**:
   - Backbone congelado para preservar features gerais
   - Apenas embedding head e classificador são treinados
   - Taxa de aprendizado reduzida para adaptação suave

### Resultados e Métricas

- Acurácia de treinamento e validação
- Matriz de confusão
- Probabilidades de classificação
- Curvas de aprendizado

## 🎯 Desafios e Aprendizados

### Dificuldades Encontradas

- Encontrar o número ideal de épocas para o treinamento foi um desafio significativo
- Técnicas de machine learning do trabalho anterior não se mostraram efetivas neste contexto, pois o escopo era completamente diferente com o uso de CNNs
- Atingir uma acurácia superior a 95% exigiu tempo e ajustes graduais, começando de aproximadamente 60%
- A adaptação de técnicas de machine learning tradicionais para o contexto de CNNs se mostrou um desafio, pois muitas abordagens que funcionavam bem no trabalho anterior acabavam piorando a acurácia neste novo contexto

### Aprendizados

- O projeto demonstrou uma melhoria significativa em relação ao trabalho anterior, que explorava as limitações da falta de CNNs
- Conseguimos atingir uma acurácia impressionante de quase 97% com um modelo relativamente pequeno (49 fotos por pessoa)
- Ficou evidente que as CNNs são fundamentais e indispensáveis para qualquer modelo de visão computacional no mercado atual
- A evolução da acurácia de 60% para quase 97% demonstrou a importância do ajuste fino e da paciência no desenvolvimento de modelos de deep learning
- O sucesso do projeto com um conjunto de dados relativamente pequeno (49 fotos por pessoa) comprovou a eficiência das CNNs em tarefas de reconhecimento facial

### Impacto no Desenvolvimento

- A experiência com este projeto reforçou a importância das CNNs em aplicações modernas de visão computacional
- O contraste entre o trabalho anterior (sem CNNs) e este projeto demonstrou claramente a superioridade das redes neurais convolucionais em tarefas de reconhecimento facial
- A jornada de desenvolvimento, desde os primeiros resultados modestos até a alta acurácia final, serviu como valiosa experiência de aprendizado sobre o processo iterativo de desenvolvimento de modelos de deep learning

## 🤝 Contribuindo

Sinta-se à vontade para abrir issues para relatar bugs ou sugerir melhorias.

## 📄 Licença

Este projeto está sob a licença MIT. 
