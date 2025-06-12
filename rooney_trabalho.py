import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import mediapipe as mp
import cv2
import numpy as np
import os
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configurações otimizadas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

# Dataset personalizado com correções
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.class_weights = []
        
        # Carregar dados e criar mapeamento de labels
        idx = 0
        class_counts = {}
        
        print(f"Procurando imagens em: {root_dir}")
        
        # Garantir ordem consistente das classes
        sorted_classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"Classes encontradas: {sorted_classes}")
        
        for label in sorted_classes:
            label_dir = os.path.join(root_dir, label)
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label
            class_counts[idx] = 0
            
            print(f"\nProcessando pasta: {label_dir}")
            
            img_files = [f for f in os.listdir(label_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Encontrados {len(img_files)} arquivos de imagem")
            
            for img_name in img_files:
                img_path = os.path.join(label_dir, img_name)
                # Remover a verificação cv2.imread aqui, pois ela falha com caracteres Unicode no caminho.
                # O carregamento real da imagem é feito em __getitem__ usando Pillow.
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.labels.append(idx)
                    class_counts[idx] += 1
                    # print(f"Imagem adicionada com sucesso: {img_name}") # Pode ser muito verboso, remova se quiser
                else:
                    print(f"AVISO: Arquivo não encontrado ou inacessível: {img_path}")
            idx += 1
        
        # Calcular pesos das classes para balanceamento
        if class_counts:
            total_samples = sum(class_counts.values())
            self.class_weights = [total_samples / (len(class_counts) * max(1, count)) 
                                for count in class_counts.values()]
        
        print(f"\nDataset carregado: {len(self.images)} imagens, {len(self.label_to_idx)} classes")
        print(f"Distribuição de classes: {dict(zip(self.idx_to_label.values(), class_counts.values()))}")
        
        if len(self.images) == 0:
            print("\nAVISO: Nenhuma imagem foi carregada! Verifique:")
            print("1. Se os caminhos das pastas estão corretos")
            print("2. Se as imagens estão nos formatos suportados (.jpg, .jpeg, .png, .bmp, .tiff)")
            print("3. Se as imagens podem ser abertas pelo OpenCV")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Usar PIL para abrir a imagem, que tem melhor suporte a Unicode em caminhos no Windows
            # e depois converter para o formato do OpenCV.
            image_pil = Image.open(img_path).convert('RGB')
            image = np.array(image_pil) # Converter de PIL para NumPy array
            
            if image is None or image.size == 0:
                raise ValueError(f"Não foi possível carregar a imagem ou imagem vazia após conversão: {img_path}")
            
            # O cv2.cvtColor(image, cv2.COLOR_BGR2RGB) não é mais necessário aqui 
            # porque o PIL já converteu para RGB.
            # Se o modelo espera BGR, pode ser necessário um ajuste futuro, 
            # mas geralmente PyTorch espera RGB.
            
            # Redimensionar para garantir tamanho consistente
            image = cv2.resize(image, (256, 256))
            
            # Pré-processamento básico
            image = self._preprocess_image(image)
            
            if self.transform:
                if self.is_training:
                    # Aplicar augmentação mais controlada
                    image = self._apply_augmentation(image)
                image = self.transform(image)
            
            return image, label, img_path  # Retornar também o caminho da imagem para predições
            
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {e}")
            # Retornar uma imagem válida aleatória
            valid_idx = random.randint(0, len(self.images)-1)
            if valid_idx != idx:
                return self.__getitem__(valid_idx)
            else:
                # Criar uma imagem dummy se necessário
                dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
                if self.transform:
                    dummy_image = self.transform(Image.fromarray(dummy_image))
                return dummy_image, label, img_path
    
    def _preprocess_image(self, image):
        """Pré-processamento controlado da imagem"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = cv2.bilateralFilter(image, 5, 50, 50)
        return image
    
    def _apply_augmentation(self, image):
        """Augmentação mais controlada"""
        pil_image = Image.fromarray(image)
        
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.9, 1.1))
            
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        image = np.array(pil_image)
        
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w//2, h//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image

# Modelo simplificado
class ImprovedFaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=EMBEDDING_SIZE, dropout_rate=DROPOUT_RATE):
        super(ImprovedFaceRecognitionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Sequential(
            nn.Linear(num_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, num_classes)
        )
        self.embedding_size = embedding_size

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding_normalized = nn.functional.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return logits, embedding_normalized

# Função para predição de todas as imagens
def predict_all_images(model, data_loader, class_names, output_file="predictions.csv"):
    """Realiza predições em todas as imagens e salva os resultados"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, img_paths) in enumerate(data_loader):
            images = images.to(DEVICE)
            try:
                logits, _ = model(images)
                probabilities = torch.softmax(logits, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                for img_path, pred_idx, conf, true_label in zip(img_paths, predicted.cpu().numpy(), 
                                                              confidences.cpu().numpy(), labels.numpy()):
                    pred_label = class_names[pred_idx]
                    predictions.append({
                        "image_path": img_path,
                        "predicted_label": pred_label,
                        "confidence": float(conf),
                        "true_label": class_names[true_label] if true_label >= 0 else "Unknown"
                    })
                
            except Exception as e:
                print(f"Erro no batch {batch_idx}: {e}")
                continue
    
    # Salvar predições em CSV
    if predictions:
        df = pd.DataFrame(predictions)
        df.to_csv(output_file, index=False)
        print(f"Predições salvas em {output_file}")
        
        # Resumo das predições
        pred_counts = df["predicted_label"].value_counts().to_dict()
        print("\nResumo das Predições:")
        for label, count in pred_counts.items():
            print(f"Classe {label}: {count} imagens")
    
    return predictions

# Função de avaliação
def evaluate_model_detailed(model, test_loader, class_names=None, plot_confusion=True):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            try:
                logits, _ = model(images)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Erro no batch {batch_idx}: {e}")
                continue
    
    if total == 0:
        print("Nenhuma amostra foi processada!")
        return 0.0
    
    accuracy = 100 * correct / total
    print(f"\nResultados da Avaliação: Acurácia: {accuracy:.2f}%")
    
    if len(all_preds) == 0 or len(all_labels) == 0:
        print("Nenhuma predição válida encontrada!")
        return accuracy
    
    if class_names is not None:
        print(f"\nRelatório de Classificação:")
        report = classification_report(all_labels, all_preds, 
                                     target_names=class_names, 
                                     zero_division=0)
        print(report)
    
    if plot_confusion and class_names is not None:
        try:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(12, 10))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Número de Amostras'}, square=True)
            plt.title(f'Matriz de Confusão\nAcurácia: {accuracy:.2f}%', fontsize=16, pad=20)
            plt.xlabel('Classe Predita', fontsize=12)
            plt.ylabel('Classe Verdadeira', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig("confusion_matrix_fixed.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Proporção'}, square=True, vmin=0, vmax=1)
            plt.title(f'Matriz de Confusão Normalizada\nAcurácia: {accuracy:.2f}%', 
                     fontsize=16, pad=20)
            plt.xlabel('Classe Predita', fontsize=12)
            plt.ylabel('Classe Verdadeira', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig("confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Erro ao plotar matriz de confusão: {e}")
    
    return accuracy

# Função de treinamento
def train_model_improved(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            try:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits, embeddings = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            except Exception as e:
                print(f"Erro no batch {batch_idx}: {e}")
                continue
        
        if total_train == 0:
            print(f"Nenhuma amostra de treino processada no epoch {epoch+1}")
            continue
            
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate_model_detailed(model, val_loader, class_names=None, plot_confusion=False)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_face_model_fixed.pt")
            print(f"Novo melhor modelo salvo! Acurácia: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping ativado. Melhor acurácia de validação: {best_val_acc:.2f}%")
            break
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
    
    if os.path.exists("best_face_model_fixed.pt"):
        model.load_state_dict(torch.load("best_face_model_fixed.pt"))
        print("Melhor modelo carregado para avaliação final.")
    
    return train_losses, val_accuracies

def run_training_pipeline(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_names, model_path="face_recognition_model_fixed.pt", mapping_path="label_mapping_fixed.json"):
    print("\n" + "="*50)
    print("INICIANDO PIPELINE DE TREINAMENTO")
    print("="*50)
    
    train_losses, val_accuracies = train_model_improved(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
    )
    
    print("\n" + "="*50)
    print("AVALIAÇÃO FINAL NO CONJUNTO DE VALIDAÇÃO")
    test_accuracy = evaluate_model_detailed(model, val_loader, class_names=class_names)
    print(f"Acurácia final na validação: {test_accuracy:.2f}%")
    
    # Realizar predições em todas as imagens (se necessário)
    # print("\n" + "="*50)
    # print("REALIZANDO PREDIÇÕES EM TODAS AS IMAGENS")
    # predict_all_images(model, combined_loader, class_names, output_file="all_predictions.csv")
    
    # Salvar modelo e mapeamento
    torch.save(model.state_dict(), model_path)
    label_mapping = {
        'label_to_idx': {name: i for i, name in enumerate(class_names)},
        'idx_to_label': {i: name for i, name in enumerate(class_names)}
    }
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"Modelo salvo como: {model_path}")
    print(f"Mapeamento salvo como: {mapping_path}")
    
    # Plotar curvas de treinamento
    if train_losses and val_accuracies:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Custom')
        plt.title('Perda de Treinamento')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'r-', label='Custom')
        plt.title('Acurácia de Validação')
        plt.xlabel('Época')
        plt.ylabel('Acurácia (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\nTreinamento concluído!")

# Função principal
def main():
    print("Iniciando sistema de reconhecimento facial corrigido...")
    
    # Verificar diretórios
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "Fotos")
    test_dir = os.path.join(base_dir, "Fotos")
    
    print(f"Diretório base: {base_dir}")
    print(f"Diretório de treino: {train_dir}")
    print(f"Diretório de teste: {test_dir}")
    
    for directory in [train_dir, test_dir]:
        if not os.path.exists(directory):
            print(f"Diretório não encontrado: {directory}")
            return
        else:
            print(f"Diretório encontrado: {directory}")
            print("Conteúdo do diretório:")
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    print(f"- Pasta: {item}")
                    print(f"  Conteúdo: {os.listdir(item_path)}")
    
    # Transformações
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Carregar datasets individuais
    print("Carregando datasets...")
    try:
        train_dataset = FaceDataset(train_dir, transform=train_transform, is_training=True)
        test_dataset = FaceDataset(test_dir, transform=val_transform, is_training=False)
        
        # Verificar se os datasets têm as mesmas classes
        if train_dataset.label_to_idx != test_dataset.label_to_idx:
            print("Erro: Os datasets têm mapeamentos de classes diferentes!")
            print(f"Train classes: {train_dataset.label_to_idx}")
            print(f"Test classes: {test_dataset.label_to_idx}")
            return
        
        # Combinar datasets
        combined_dataset = ConcatDataset([train_dataset, test_dataset])
        print(f"Dataset combinado: {len(combined_dataset)} imagens")
        
        if len(combined_dataset) == 0:
            print("Dataset combinado está vazio!")
            return
        
        # Dividir em treino e validação
        train_indices = []
        val_indices = []
        labels = [train_dataset.labels[idx] if idx < len(train_dataset) else test_dataset.labels[idx - len(train_dataset)] 
                 for idx in range(len(combined_dataset))]
        
        for class_idx in range(len(train_dataset.label_to_idx)):
            indices = [i for i, label in enumerate(labels) if label == class_idx]
            random.seed(42)
            random.shuffle(indices)
            n_samples = len(indices)
            val_count = max(1, int(0.25 * n_samples))
            train_count = n_samples - val_count
            train_indices.extend(indices[:train_count])
            val_indices.extend(indices[train_count:])
            class_name = train_dataset.idx_to_label[class_idx]
            print(f"Classe {class_name}: {train_count} treino, {val_count} validação")
        
        train_subset = torch.utils.data.Subset(combined_dataset, train_indices)
        val_subset = torch.utils.data.Subset(combined_dataset, val_indices)
        
        # Aplicar transformação de validação
        def set_val_transform(dataset):
            if isinstance(dataset, torch.utils.data.Subset):
                dataset.dataset.datasets[0].transform = val_transform
                dataset.dataset.datasets[0].is_training = False
                dataset.dataset.datasets[1].transform = val_transform
                dataset.dataset.datasets[1].is_training = False
        
        set_val_transform(val_subset)
        
        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)
        # combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        #                            num_workers=2, pin_memory=True)
        
        # Obter nomes das classes
        class_names = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.label_to_idx))]
        print(f"Classes encontradas: {class_names}")
        
        # Inicializar modelo
        num_classes = len(train_dataset.label_to_idx)
        print(f"Inicializando modelo com {num_classes} classes...")
        model = ImprovedFaceRecognitionModel(num_classes=num_classes).to(DEVICE)
        
        # Garantir que o modelo esteja em float32 para evitar erros de tipo
        model.float()
        
        # Critério e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
        
        # Treinar modelo
        run_training_pipeline(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, class_names)
        
        # Salvar modelo e mapeamento no final do treinamento inicial, já feito por run_training_pipeline
        # torch.save(model.state_dict(), "face_recognition_model_fixed.pt")
        # label_mapping = {
        #     'label_to_idx': train_dataset.label_to_idx,
        #     'idx_to_label': train_dataset.idx_to_label
        # }
        # with open("label_mapping_fixed.json", "w") as f:
        #     json.dump(label_mapping, f, indent=2)
        
        # Plotar curvas de treinamento (já feito por run_training_pipeline)
        # if train_losses and val_accuracies:
        #     plt.figure(figsize=(12, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.plot(train_losses, 'b-', label='Custom')
        #     plt.title('Perda de Treinamento')
        #     plt.xlabel('Época')
        #     plt.ylabel('Perda')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.subplot(1, 2, 2)
        #     plt.plot(val_accuracies, 'r-', label='Custom')
        #     plt.title('Acurácia de Validação')
        #     plt.xlabel('Época')
        #     plt.ylabel('Acurácia (%)')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        
        print("\nTreinamento e predição concluídos!")
        print(f"Modelo salvo como: face_recognition_model_fixed.pt")
        print(f"Mapeamento salvo como: label_mapping_fixed.json")
        # print(f"Predições salvas como: all_predictions.csv")
        
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()

def add_new_user_and_finetune(new_user_name, new_user_image_paths):
    print("\n" + "="*50)
    print(f"ADICIONANDO NOVO USUÁRIO: {new_user_name.upper()}")
    print("="*50)

    # 1. Carregar mapeamento de labels existente
    label_mapping = {}
    if os.path.exists("label_mapping_fixed.json"):
        with open("label_mapping_fixed.json", "r") as f:
            label_mapping = json.load(f)
        print(f"Mapeamento de labels existente carregado: {label_mapping}")
    
    current_label_to_idx = {k: int(v) for k, v in label_mapping.get('label_to_idx', {}).items()}
    current_idx_to_label = {int(k): v for k, v in label_mapping.get('idx_to_label', {}).items()}
    
    num_existing_classes = len(current_label_to_idx)

    # 2. Preparar diretório para o novo usuário e copiar imagens
    base_dir = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(base_dir, "Fotos")
    new_user_dir = os.path.join(photos_dir, new_user_name)
    os.makedirs(new_user_dir, exist_ok=True)
    print(f"Criado/verificado diretório para {new_user_name}: {new_user_dir}")

    copied_images_count = 0
    for i, img_path_src in enumerate(new_user_image_paths):
        # Gerar um nome de arquivo único para evitar colisões
        ext = os.path.splitext(img_path_src)[1]
        img_name_dest = f"{new_user_name}_{i+1}{ext}"
        img_path_dest = os.path.join(new_user_dir, img_name_dest)
        try:
            # Usar Pillow para re-salvar a imagem para garantir compatibilidade e mover
            Image.open(img_path_src).save(img_path_dest)
            print(f"Imagem '{os.path.basename(img_path_src)}' copiada para '{new_user_dir}'")
            copied_images_count += 1
        except Exception as e:
            print(f"AVISO: Não foi possível copiar a imagem {img_path_src}: {e}")

    if copied_images_count == 0:
        print("ERRO: Nenhuma imagem válida fornecida para o novo usuário.")
        return

    # 3. Carregar datasets atualizados
    print("Carregando datasets atualizados...")
    # Usar o mesmo diretório 'Fotos' para treino e teste, pois é onde todas as classes estão
    base_dir = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(base_dir, "Fotos")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # O FaceDataset agora reconstruirá a lista de classes com o novo usuário
    full_dataset = FaceDataset(photos_dir, transform=train_transform, is_training=True) # Usar train_transform para o dataset completo
    updated_class_names = [full_dataset.idx_to_label[i] for i in range(len(full_dataset.label_to_idx))]
    print(f"Classes atualizadas no dataset: {updated_class_names}")

    if new_user_name not in updated_class_names:
        print(f"ERRO: Novo usuário '{new_user_name}' não foi detectado no dataset. Verifique as imagens fornecidas.")
        return

    # 4. Dividir em treino e validação com as novas classes
    train_indices = []
    val_indices = []
    labels = full_dataset.labels

    for class_idx in range(len(full_dataset.label_to_idx)):
        indices = [i for i, label in enumerate(labels) if label == class_idx]
        random.seed(42) # Usar uma seed fixa para reprodutibilidade
        random.shuffle(indices)
        n_samples = len(indices)
        val_count = max(1, int(0.25 * n_samples))
        train_count = n_samples - val_count
        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count:])
        class_name = full_dataset.idx_to_label[class_idx]
        print(f"Classe {class_name}: {train_count} treino, {val_count} validação (total {n_samples})")
    
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # Aplicar transformação de validação ao subset de validação
    def set_val_transform(dataset_subset):
        if isinstance(dataset_subset, torch.utils.data.Subset):
            # Certifique-se de que a transformação correta é aplicada para os dados de validação
            # Isso pode exigir mais granularidade se o dataset subjacente for ConcatDataset, etc.
            # Por simplicidade, vamos assumir que o dataset subjacente é o FaceDataset original
            dataset_subset.dataset.transform = val_transform
            dataset_subset.dataset.is_training = False
    
    set_val_transform(val_subset)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, 
                       num_workers=2, pin_memory=True)

    # 5. Carregar e adaptar o modelo
    model = ImprovedFaceRecognitionModel(num_classes=num_existing_classes).to(DEVICE) # Carregar com num de classes antigas
    model.float() # Garantir float32
    if os.path.exists("face_recognition_model_fixed.pt"):
        model.load_state_dict(torch.load("face_recognition_model_fixed.pt", map_location=DEVICE))
        print("Modelo pré-existente carregado para fine-tuning.")
    else:
        print("AVISO: Nenhum modelo pré-existente encontrado. Iniciando treinamento do zero.")

    # Adaptar a camada final se o número de classes mudou
    num_current_classes = len(updated_class_names)
    if model.classifier[0].out_features != num_current_classes:
        print(f"Número de classes mudou de {model.classifier[0].out_features} para {num_current_classes}.")
        # Substituir a camada classificadora
        new_classifier = nn.Sequential(
            nn.Linear(model.embedding_size, num_current_classes)
        ).to(DEVICE) # Mover para o dispositivo
        model.classifier = new_classifier
        print("Camada classificadora adaptada para novas classes.")
        # Inicializar pesos da nova camada para evitar valores muito grandes/pequenos
        nn.init.xavier_uniform_(model.classifier[0].weight)
        model.classifier[0].bias.data.fill_(0.01)
    
    # Congelar o backbone para fine-tuning leve
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone congelado para fine-tuning.")

    # Descongelar apenas as camadas do embedding head e o novo classificador
    for param in model.embedding_head.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Embedding head e classificador descongelados.")

    # Otimizador para fine-tuning (pode usar uma taxa de aprendizado menor)
    finetune_learning_rate = LEARNING_RATE * 0.1 # Exemplo: 1/10 da LR original
    finetune_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                       lr=finetune_learning_rate, weight_decay=WEIGHT_DECAY)
    finetune_scheduler = ReduceLROnPlateau(finetune_optimizer, mode='max', factor=0.5, patience=3) # Paciência menor para finetune rápido

    # Critério de perda
    criterion = nn.CrossEntropyLoss()
    
    # 6. Rodar o pipeline de treinamento para fine-tuning
    finetune_epochs = max(3, int(EPOCHS / 2)) # Ex: 3 a 5 épocas para finetune
    print(f"Iniciando fine-tuning por {finetune_epochs} épocas...")
    run_training_pipeline(model, train_loader, val_loader, criterion, finetune_optimizer, 
                            finetune_scheduler, finetune_epochs, 
                            updated_class_names, 
                            model_path="face_recognition_model_fixed.pt", # Salvar no mesmo lugar
                            mapping_path="label_mapping_fixed.json")
    print("\nFine-tuning concluído!")

if __name__ == "__main__":
    main()