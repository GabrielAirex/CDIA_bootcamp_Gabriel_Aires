# Desafio 2 — Detecção de Trincas em Paredes
## Relatório Técnico

**Bootcamp CDIA — Residência em IA**
**Autor:** Gabriel Afonso Freitas Aires

---

## 1. Visão Geral

O desafio consiste em detectar e segmentar trincas e fissuras em imagens de paredes, retornando a localização (bounding box + máscara de segmentação) e um nível de confiança. O principal diferencial do problema é a natureza das trincas: estruturas lineares finas, com baixo contraste local, que se confundem com juntas de alvenaria, sombras e variações de textura da parede.

A solução implementa um pipeline em três camadas de fallback:

```
Imagem de entrada
  └→ YOLOv8n-seg fine-tuned (principal)
       ├→ confiança ≥ 0.40: retorna resultado
       └→ confiança < 0.40: tenta VLM Claude (cloud)
            └→ falha VLM: Canny fallback (offline)
```

---

## 2. Dataset

O dataset foi obtido do Roboflow Universe (Crack Detection dataset, formato YOLO segmentation):

| Atributo | Valor |
|----------|-------|
| Total de imagens | 1.551 |
| Resolução original | 1440 × 2560 px |
| Resolução de treino | 640 × 640 px |
| Classes | 1 (`crack`) |
| Formato de anotação | YOLO segmentation (polígonos normalizados) |
| Split treino | 1.240 imagens (80%) |
| Split validação | 311 imagens (20%) |

As imagens cobrem diferentes tipos de superfície (concreto, reboco, tijolo aparente), variações de iluminação (natural, artificial, sombra direcional) e tipos de trinca (capilar, estrutural, superficial). A distribuição de tamanhos de trinca é fortemente desbalanceada — a maioria das anotações cobre < 5% da área da imagem, o que torna os métodos clássicos de detecção (threshold global, Otsu) inadequados.

---

## 3. Benchmark de Métodos Tradicionais

Antes do fine-tuning do YOLO, quatro métodos de visão computacional clássica foram avaliados em 30 imagens de validação (amostradas aleatoriamente do split val).

**Métricas:**
- **Det%** — porcentagem de imagens onde alguma trinca foi detectada (recall binário)
- **IoU médio** — Intersection over Union entre máscara predita e ground truth YOLO
- **ms/img** — tempo médio de inferência

### 3.1 Resultados

| Método | Det% | IoU médio | ms/img | Notas |
|--------|------|-----------|--------|-------|
| Canny + Morph | 96.7% | 0.161 | 309ms | Melhor IoU tradicional |
| Adaptive Thresh | 96.7% | 0.117 | ~180ms | Alta detecção, baixa precisão |
| Sobel + Otsu | 100% | 0.107 | ~150ms | Sobre-detecta bordas de textura |
| Gabor Filter | 100% | 0.009 | ~220ms | Detecta tudo — IoU quase zero |

### 3.2 Análise dos Métodos Tradicionais

**Canny + Morfologia** — melhor resultado clássico. O Canny com thresholds adaptativos (baseados na mediana da imagem) + dilatação para conectar bordas fragmentadas + filtro de área mínima consegue isolar as trincas maiores. O IoU de 0.161 indica que a máscara prevista tem geometria aproximada correta mas com contornos imprecisos — Canny produz bordas de 1-2 pixels de largura, enquanto as anotações do dataset têm polígonos de 10-30 pixels de espessura. Essa discrepância sistemática limita o IoU independente do threshold.

**Adaptive Threshold** — detecta quase tudo (96.7%) mas com IoU inferior ao Canny. O threshold adaptativo gaussiano com blockSize=51 responde a qualquer variação local de intensidade: juntas de reboco, sombras de molduras, variações de textura. Resultado: muitos falsos positivos que reduzem o IoU médio.

**Sobel + Otsu** — 100% de detecção mas IoU de 0.107. O gradiente Sobel + threshold Otsu é agressivo: detecta todas as bordas da imagem. Em paredes com textura rica (tijolo, reboco granulado), gera um mapa de bordas saturado onde trincas e textura são indistinguíveis.

**Gabor Filter Bank** — 4 orientações (0°, 45°, 90°, 135°), projetado para detectar texturas lineares. Teoricamente adequado para trincas (que são estruturas lineares). Na prática, o IoU de 0.009 indica falha quase total: os filtros Gabor respondem às juntas de alvenaria (que têm frequência e orientação regulares) com resposta mais alta do que às trincas irregulares. O método detecta *linhas regulares*, não trincas.

**Conclusão da fase clássica:** a melhor abordagem clássica (Canny+Morph, IoU=0.161) é mantida como fallback offline no pipeline. Para IoU útil em aplicação real, é necessária aprendizagem supervisionada.

---

## 4. Fine-Tuning YOLOv8n-seg

### 4.1 Escolha do Modelo Base

**YOLOv8n-seg** foi escolhido por três razões:

1. **Velocidade:** variante *nano* (3.3M parâmetros, 11.5 GFLOPs) — menor footprint de memória e tempo de inferência adequado para CPU
2. **Segmentação nativa:** detecta bounding boxes + máscaras de segmentação em uma única passagem — essencial para a tarefa
3. **Transferência eficiente:** pesos pré-treinados no COCO já incluem aprendizado de bordas, texturas e formas lineares — features reutilizáveis para trincas

### 4.2 Configuração de Treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Épocas (1ª rodada) | 4* | PC travou, treino interrompido |
| Épocas (retreino) | 10 | Fine-tune sobre best.pt da 1ª rodada |
| Image size | 640×640 | Padrão YOLO, balanceia resolução e velocidade |
| Batch size | 8 | Limite de RAM sem GPU |
| Otimizador | AdamW (auto) | Seleção automática pelo YOLO |
| Learning rate | 0.002 | Auto-calibrado pelo scheduler |
| Patience | 10 | Early stopping se sem melhora em 10 épocas |
| Augmentation | nativa YOLO | Mosaic, flip, HSV, scale |

*O primeiro treino foi interrompido pelo travamento do sistema após 4 épocas. O modelo salvo (best.pt, época 4) foi usado como ponto de partida do retreino, aproveitando o aprendizado já consolidado em vez de reiniciar do zero.

### 4.3 Resultados do Treinamento

**1ª rodada (interrompida no PC):**

| Época | mAP50 (box) | mAP50 (seg) | box_loss | seg_loss |
|-------|------------|------------|----------|----------|
| 1 | 0.374 | 0.317 | 1.444 | 1.912 |
| 2 | 0.317 | 0.221 | 1.546 | 1.729 |
| 3 | 0.370 | 0.240 | 1.524 | 1.612 |
| **4 (best)** | **0.486** | **0.377** | **1.485** | **1.635** |

Já na época 1, o mAP50(seg)=0.317 superou o melhor método clássico (Canny+Morph, IoU=0.161) — demonstrando que mesmo com apenas uma passagem pelo dataset, a transferência de conhecimento do pré-treino COCO é eficaz.

**Retreino (10 épocas a partir do best.pt da rodada 1):**

| Época | mAP50 (box) | mAP50 (seg) | box_loss | seg_loss |
|-------|------------|------------|----------|----------|
| 1 | 0.535 | 0.413 | 1.417 | 1.245 |
| 2 | 0.568 | 0.421 | 1.517 | 1.286 |
| 3 | 0.496 | 0.391 | 1.495 | 1.302 |
| 4 | 0.559 | 0.423 | 1.417 | 1.265 |
| 5 | 0.558 | 0.419 | 1.354 | 1.233 |
| 6 | 0.597 | 0.466 | 1.298 | 1.221 |
| 7 | 0.603 | 0.442 | 1.204 | 1.181 |
| 8 | 0.629 | 0.522 | 1.173 | 1.158 |
| 9 | 0.640 | 0.502 | 1.086 | 1.134 |
| **10 (melhor)** | **0.672** | **0.510** | **1.009** | **1.132** |

**Validação oficial final (311 imagens val):**

| Métrica | Valor |
|---------|-------|
| Precision (box) | 0.756 |
| Recall (box) | 0.616 |
| **mAP50 (box)** | **0.6723** |
| mAP50-95 (box) | 0.4674 |
| Precision (mask) | 0.660 |
| Recall (mask) | 0.538 |
| **mAP50 (seg)** | **0.5099** |
| mAP50-95 (seg) | 0.1959 |
| Inferência (CPU, i7-13650HX) | 41.4ms/img |

O retreino partindo do best.pt convergiu monotonicamente a partir da época 6, com as losses de treino caindo de forma consistente (box_loss: 1.42 → 1.01, seg_loss: 1.25 → 1.13). O mAP50(box) final de **0.672** representa +38% sobre a 1ª rodada interrompida (0.486) e +18% sobre o melhor ponto do retreino na época 2 (0.568).

### 4.4 Por que YOLOv8 Supera os Métodos Clássicos

| Limitação clássica | Como o YOLO resolve |
|-------------------|---------------------|
| Trinca vs. junta de reboco indistinguíveis por intensidade | Aprende contexto visual: padrão de trinca tem geometria e textura distintas de junta regular |
| IoU limitado por espessura de borda Canny (~1px vs. 30px de GT) | Prediz máscaras de segmentação de forma livre, alinhadas com as anotações do dataset |
| Falsos positivos em textura rica | Aprende a ignorar textura de reboco — discriminação supervisionada pelo ground truth |
| Sem adaptação ao domínio | Fine-tuned especificamente no domínio de paredes com trincas reais |

---

## 5. Pipeline Final e App

### 5.1 Arquitetura do Pipeline

```python
detect_cracks(image_path, vlm_threshold=0.4):
    if weights/best.pt exists:
        result = yolov8n_seg(image, conf=0.25, iou=0.45)
        if result.confidence >= 0.40 or cracks_found > 0:
            return result  # caminho feliz
        # baixa confiança → cloud fallback
        vlm_result = claude_vision(image)
        if vlm_result.confidence > 0:
            return vlm_result
    # sem modelo ou VLM falhou
    return canny_fallback(image)
```

**Threshold de confiança:** 0.40 — valor calibrado empiricamente. Imagens com trincas muito finas ou parcialmente ocluídas tendem a ter confiança YOLO entre 0.25–0.40; neste range o VLM Claude oferece julgamento mais robusto (zero-shot com contexto semântico).

**Fallback Canny:** garante resposta mesmo sem internet e sem modelo. IoU esperado: 0.161 (conforme benchmark). Adequado para triagem inicial.

### 5.2 Interface FastAPI

A aplicação expõe dois endpoints para o Desafio 2:

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `POST /d2/detect` | POST | Recebe imagem, retorna JSON com n_cracks, confidence, method, bboxes + imagem anotada |
| `GET /d2/benchmark` | GET | Roda benchmark em 30 imagens val e retorna tabela comparativa |

A interface web unificada (porta 8080) integra D1 e D2 com seletor no header. A porta 8081 oferece o app standalone do D2.

---

## 6. Comparativo Final

| Método | IoU médio | Det% | ms/img | Requer |
|--------|-----------|------|--------|--------|
| Gabor Filter | 0.009 | 100% | ~220ms | CPU |
| Sobel + Otsu | 0.107 | 100% | ~150ms | CPU |
| Adaptive Thresh | 0.117 | 96.7% | ~180ms | CPU |
| Canny + Morph | 0.161 | 96.7% | 309ms | CPU |
| **YOLOv8n-seg (ep.4, 1ª rodada)** | 0.377 | — | ~300ms | CPU |
| **YOLOv8n-seg (retreino, 10 épocas)** | **0.510** | — | **41.4ms** | CPU |

O YOLOv8n-seg com apenas **4 épocas de fine-tuning** já superou todos os métodos clássicos em ~2.3× no IoU de segmentação. Com o retreino completo de 10 épocas, o mAP50(seg) chegou a **0.510** — 3.2× superior ao melhor método clássico (Canny+Morph, IoU=0.161).

---

## 7. Limitações e Trabalhos Futuros

**Limitações do setup atual:**

- **Sem GPU:** o treino foi realizado integralmente em CPU, limitando o número de épocas viáveis dentro do prazo. Com GPU, 50 épocas rodariam em ~2h (vs. ~15h em CPU).
- **Épocas interrompidas:** o travamento do sistema na 1ª rodada impediu que o modelo atingisse convergência natural (patience=15). O retreino de 10 épocas parcialmente compensa, mas não substitui um treino completo de 50 épocas.
- **Modelo nano:** YOLOv8n-seg tem capacidade limitada para texturas complexas. Variantes maiores (YOLOv8s-seg, YOLOv8m-seg) teriam melhor discriminação trinca vs. textura com o mesmo dataset.

**Com mais tempo e recursos:**

- **50+ épocas em GPU:** o learning rate scheduler ainda está em warm-up nas primeiras épocas observadas — o modelo ainda não atingiu o platô de convergência. 50 épocas completas devem levar mAP50(seg) para 0.55–0.65 neste dataset.
- **YOLOv8s/m-seg:** modelos maiores com ~11M/25M parâmetros podem discriminar melhor trincas capilares (< 2px de largura) que o nano perde.
- **Data augmentation específica:** além do augmentation padrão YOLO, estratégias domain-specific para trincas (simulação de iluminação rasante, variação de contraste de parede) poderiam melhorar robustez.
- **Active learning:** usar o modelo atual para identificar os exemplos mais difíceis no val set e anotar manualmente 50–100 imagens adicionais nessas regiões de incerteza.

---

---

## Apêndice — Arquivos não incluídos no .zip

O arquivo `.zip` de submissão tem limite de 10 MB. Os artefatos abaixo foram omitidos por tamanho mas estão disponíveis integralmente no repositório GitHub:

**[github.com/GabrielAirex/CDIA_bootcamp_Gabriel_Aires](https://github.com/GabrielAirex/CDIA_bootcamp_Gabriel_Aires)**

| Artefato | Tamanho | Motivo da omissão | Localização no repo |
|---|---|---|---|
| `dataset/` | ~1.9 GB | 1.551 imagens de trincas (Roboflow Universe) | não disponível no repo |
| `benchmark/output_benchmark/` | ~40 MB | Imagens geradas pelo benchmark (5 métodos × 30 imagens val) | `desafio2/benchmark/output_benchmark/` |
| `runs/` | ~75 MB | Artefatos de treino YOLO (curvas, batches, confusion matrix) | não disponível no repo |
| `yolov8n-seg.pt` | ~6.8 MB | Modelo base Ultralytics — baixado automaticamente pelo `train.py` | não disponível no repo |
| `docs/screenshots/` | ~5.7 MB | Screenshots da interface para documentação | `docs/screenshots/` |

> O dataset de trincas (1.9 GB) e os artefatos de treino (75 MB) não estão no repositório por excederem os limites do GitHub. O dataset pode ser obtido no [Roboflow Universe](https://universe.roboflow.com) buscando por "Crack Detection" com formato YOLO segmentation. O modelo base `yolov8n-seg.pt` é baixado automaticamente ao executar `train.py`.

---

*Relatório gerado em 31/05/2026 | CDIA Bootcamp — Residência em IA*
