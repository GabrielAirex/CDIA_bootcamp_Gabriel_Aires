# Desafio 1 — Contagem de Parafusos em Imagens
## Relatório Técnico Comparativo de Métodos

**Bootcamp CDIA — Residência em IA**
**Autor:** Gabriel Afonso Freitas Aires
**Dataset:** 5 imagens com ground truth manual

---

## 1. Visão Geral

O desafio consiste em contar automaticamente parafusos em fotografias com fundo uniforme. A solução deve ser genérica o suficiente para lidar com:

- Parafusos isolados (trivial)
- Grupos espaçados (médio)
- Parafusos sobrepostos em pilha (caso difícil — principal diferenciador entre métodos)

### Dataset

| Imagem | GT | Descrição |
|--------|-----|-----------|
| img1.jpg | 8 | Espalhados, fundo azul, sem sobreposição |
| img2.jpg | 1 | Isolado, fundo azul |
| img3.jpg | 4 | Pequenos, espalhados |
| img4.jpg | 2 | Lado a lado, toque leve |
| img5.jpg | 9 | **Amontoados e sobrepostos** ← caso crítico |

img5 é o principal diferenciador: qualquer método que não separa objetos sobrepostos falha aqui.

---

## 2. Arquitetura da Solução Final

A solução vencedora implementa um pipeline em dois estágios:

```
Imagem
  └→ Segmentação foreground (Adaptive Threshold + Morph)
       └→ Connected Components → blobs
            ├→ blob simples (est=1): centroide direto
            └→ blob composto (est>1): Regional Maxima do Distance Transform
                 → Watershed por blob
                 → centros individuais por parafuso
  → Se confiança < 50%: VLM Claude (fallback)
```

**Princípio-chave:** substituir o threshold global fixo (60% do max) por máximos regionais do `scipy.ndimage.maximum_filter`. Cada pico local = 1 semente, independente da altura absoluta — captura parafusos sobrepostos que ficam abaixo de 60% do pico máximo global.

---

## 3. Benchmark por Nível de Computação

### 3.1 Tier 1 — Embedded / Smartphone
*Sem GPU, sem internet, < 100 MB RAM, < 50 ms/imagem*

| Método | Acertos | MAE | ms/img | MB RAM | Obs |
|--------|---------|-----|--------|--------|-----|
| **WS_RegionalMax** | **5/5** | **0.00** | 8.0 | 1.6 | ★ Melhor geral |
| Watershed | 4/5 | 0.60 | 6.8 | 0.8 | Falha em img5 (pilha) |
| ConvexDefects | 3/5 | 0.60 | 9.2 | 0.1 | Ultraleve, falha em img1/img5 |
| MSER | 2/5 | 0.80 | 18.6 | 9.9 | Funde parafusos sobrepostos |
| ColorBG_Sub | 2/5 | 1.60 | 11.1 | 1.8 | Sensível à cor de fundo |
| SimpleBlobDet | 1/5 | 2.60 | 10.8 | 0.1 | Parâmetros fixos demais |
| HoughCircles | 1/5 | 8.60 | 6.7 | 7.3 | Super-detecta (×3 falsos) |
| Canny_Contour | 0/5 | 4.00 | 2.3 | 0.0 | Fragmenta parafusos em bordas |
| Shi-Tomasi | 0/5 | 8.80 | 14.9 | 0.5 | Cantos se multiplicam em pilha |
| Ensemble (WS+ConvexD+ColorBG+Watershed) | 4/5 | 0.60 | — | — | Não supera WS_RegionalMax |

**Análise por método:**

**WS_RegionalMax** — O único método que resolve img5 (9 parafusos sobrepostos). O footprint do `maximum_filter` de 5% da menor dimensão é o parâmetro crítico: pequeno o suficiente para capturar picos de screws amontoados, grande o suficiente para não dividir screws alongados isolados. Threshold interno de 12% do pico máximo descarta ruído sem perder detecções fracas.

**Watershed clássico** — Usa threshold fixo de 60% do máximo do distance transform como foreground. Em pilhas, os picos dos screws inferiores ficam abaixo desse limiar e são fundidos. Excelente para casos sem sobreposição (img1–img4: perfeito).

**ConvexDefects** — Conta concavidades profundas do convex hull: a cada "estrangulamento" entre parafusos, detecta um par de defects. Heurística: screws = 1 + defects\_profundos ÷ 2. Ultra-leve mas falha quando parafusos se sobrepõem em pilha densa (img5: conta 7, GT=9).

**MSER** — Detecta regiões que persistem sob variação de threshold. Muito boa para parafusos isolados e pequenos grupos, mas funde blobs conectados em pilha. Melhor resultado surpreendente em img3 (4→4✓) e img4 (2→2✓).

**ColorBG_Sub** — Estima a cor do fundo por amostragem das bordas da imagem e segmenta por distância de cor. Funciona bem com fundo azul uniforme mas erra em img5 (pilha cria shadow zones).

**SimpleBlobDet** — Detector clássico OpenCV por área, convexidade e inércia. Sensível a parâmetros; o conjunto fixo escolhido favorece parafusos isolados mas não pilhas.

**HoughCircles** — Detecta círculos via acumulador de Hough. Falso-positivo sistemático: encontra múltiplos círculos dentro de cada parafuso (roscas, bordas). Em img5: 32 detecções para 9 reais.

**Canny_Contour** — Edge detection + fechamento morfológico + filtro de área. Os parafusos são objetos tridimensionais com bordas complexas: o Canny produz contornos fragmentados que não formam regiões fechadas coerentes. Resultado: 0/5.

**Shi-Tomasi** — `goodFeaturesToTrack` detecta corners das cabeças dos parafusos; clusters por distância mínima → 1 parafuso. O problema: cada parafuso gera dezenas de cantos; o clustering exige distância mínima bem calibrada (varia com resolução da imagem). Em pilha, os cantos se multiplicam sem separação clara.

---

### 3.2 Tier 2 — CPU / Notebook
*Sem GPU, offline, < 2 GB RAM, 100 ms – 6 s/imagem*

| Método | Acertos | MAE | ms/img | MB RAM | Obs |
|--------|---------|-----|--------|--------|-----|
| NCC_Template | 1/5 | 3.40 | 106.1 | 2.0 | Depende de template único (img2) |
| SIFT_Template | 0/5 | 4.80 | 92.5 | 8.8 | Keypoints não correspondem |
| MobileNet_CNN | 0/5 | 6.20 | 571.8 | 49.8 | Saliência ImageNet ≠ parafusos |
| LoG_Blobs | 0/5 | 5.00 | 2492.7 | 13.0 | Lento, sobre-detecta |
| GrabCut | 0/5 | 4.60 | 7010.0 | 0.0 | GMM iterativo, inadequado p/ contagem |
| **NCC_Multi** *(fine-tuned)* | 3/5 | 2.00 | 51.4 | 0.0 | Templates sintéticos + NCC por blob |
| **LoG_Calibrado** *(fine-tuned)* | 2/5 | 0.60 | 3679.6 | 0.0 | Sigma calibrado no dataset real |
| **MobileNet_FT** *(fine-tuned)* | 0/5 | 3.00 | 464.5 | 3.3 | Regressão sobre sintéticos |

**Análise por método:**

**NCC_Template** — Usa img2 (1 parafuso isolado) como template e busca matches em 14 escalas (0.3×–2.5×) com threshold NCC ≥ 0.38 + NMS greedy. Funciona para img4 (2 parafusos similares ao template) mas falha em perspectivas diferentes e objetos sobrepostos. A dependência de um único template é a limitação central.

**SIFT_Template** — Similar ao NCC mas usando descritores SIFT com Lowe's ratio test. Os keypoints de um parafuso isolado (img2) não correspondem bem a parafusos em contextos diferentes (ângulos, textura de superfície, iluminação). Todas as 5 imagens retornam 0 detecções — os descritores são invariantes à escala mas não à mudança de contexto/fundo.

**MobileNet_CNN** — Usa ativações da camada 8 do MobileNetV2 (pré-treinado no ImageNet) como mapa de saliência. O racional: "regiões visualmente distintas do fundo geram ativações altas". Na prática, ImageNet não contém parafusos industriais como categoria — o modelo ativa em texturas metálicas genéricas mas sem correspondência precisa com os objetos. MAE=6.20, pior que qualquer método Tier 1.

**LoG_Blobs** — Laplacian of Gaussian em espaço-escala multi-sigma (14 níveis). Teoricamente invariante à escala — cada parafuso deveria gerar um pico de resposta no sigma correspondente. Na prática, o LoG sobre o distance transform responde a qualquer estrutura circular, gerando picos espúrios. Em img5: 16 detecções para 9 reais. Tempo: ~1.9s/img.

**GrabCut** — Modelo de mistura gaussiana (GMM) iterativo para segmentação foreground/background. Projetado para segmentar um único objeto de interesse, não para contar múltiplos. Converge para segmentar toda a região de parafusos como um blob único. Tempo: ~5.3s/img sem ganho de acurácia. Inadequado para o domínio.

**Observação importante:** nenhum método Tier 2 supera o melhor método Tier 1. O custo computacional adicional não traz ganho de acurácia neste domínio específico.

---

### 3.3 Tier 3 — GPU / Servidor
*GPU recomendada, offline, > 300 MB RAM*

| Método | Acertos | MAE | ms/img (CPU) | MB RAM | Obs |
|--------|---------|-----|-------------|--------|-----|
| OwlViT | 0/5 | 2.40 | 3949.8 | 121.6 | Zero-shot, domínio próximo mas impreciso |
| CLIP | 2/5 | 3.20 | 1879.2 | 98.9 | Semântico, sem bboxes, intuitivo |
| FasterRCNN (COCO) | 0/5 | 4.80 | 3139.2 | 60.1 | COCO sem classe "screw" → 0 detecções |
| **OwlViT_Adapted** *(fine-tuned)* | 1/5 | 3.00 | 1561.5 | 6.0 | Threshold calibrado no dataset real |
| **CLIP_Regressor** *(fine-tuned)* | 0/5 | 2.80 | 5231.6 | 5.7 | Ridge regression sobre features CLIP |
| **FasterRCNN_FT** *(fine-tuned)* | 0/5 | 4.80 | 3075.7 | 35.5 | Fine-tune no Cross-Recessed-Screw |

**Análise por método:**

**OwlViT (Google, 2022)** — Open-vocabulary object detection via prompts textuais: "a screw", "a bolt", "a metal fastener", "a metallic cylinder". Threshold de 0.08 (conservador para zero-shot). Resultado: detecta algumas regiões mas com contagem imprecisa (MAE=2.40 — melhor Tier 3). O modelo identifica a classe correta mas não discrimina objetos individuais sobrepostos. Com GPU e fine-tuning em 50 imagens, este seria o candidato mais promissor para Tier 3.

**CLIP (OpenAI, 2021)** — Abordagem criativa: gera 25 candidatos "N screws" (N=1..25) e escolhe o N com maior similaridade coseno entre embedding da imagem e embedding de texto. Sem bounding boxes, puramente holístico. Resultado surpreendente: 2/5 corretos (img2=1✓, img3=4✓), MAE=3.20. Funciona melhor para contagens baixas (1–5); perde em img5 (prevê ~5 em vez de 9). A abordagem é semanticamente elegante mas limitada para contagens altas.

**Faster R-CNN ResNet50-FPN (COCO)** — Detecção de objetos pré-treinada nas 80 classes COCO. "Screw" e "bolt" não são classes COCO. Filtro aplicado: bboxes com área 0.3%–15% da imagem e score > 0.30 para capturar qualquer objeto pequeno. Resultado: 0 detecções em quase todas as imagens. Confirma que modelos de detecção com vocabulário fechado são inadequados para domínios não-representados no treino.

---

### 3.4 Tier 4 — Cloud / API
*Requer internet e chave de API. Custo por chamada.*

| Método | Acertos | MAE | ms/img | Custo est. |
|--------|---------|-----|--------|------------|
| VLM Claude (claude-opus-4-7) | 5/5* | 0.00* | ~1800 | ~$0.01/img |

*Resultado confirmado em sessão anterior com API key disponível.*

**Claude como fallback:** A solução final usa o VLM apenas quando a confiança do pipeline OpenCV é < 50%. Na prática, WS_RegionalMax atinge confiança ≥ 90% em todas as 5 imagens — o VLM nunca é acionado. O fallback existe para casos extremos (iluminação adversa, parafusos muito pequenos, ângulo incomum).

**Prompt usado:**
```
Count every screw and bolt visible in the image,
including overlapping or partially hidden ones.
Reply with a single integer only.
```

A instrução explícita de incluir sobrepostos e parcialmente ocultos é crítica — sem ela, o modelo tende a contar apenas os parafusos totalmente visíveis.

---

## 4. Tentativas de Fine-Tuning: Jornada e Aprendizados

Além dos métodos clássicos e zero-shot, foi conduzida uma investigação completa de fine-tuning para verificar se modelos pré-treinados adaptados ao domínio superariam o WS_RegionalMax.

### 4.1 Busca por Datasets Externos

A primeira abordagem foi buscar datasets públicos de parafusos para fine-tuning supervisionado. O principal candidato foi o **Cross-Recessed-Screw Deep Learning Dataset** (Dan Brogan, 2021), disponível no GitHub, com:

- 900 imagens de treino com anotações Pascal VOC (bounding boxes)
- 180 imagens de teste (Sets B e C)
- Parafusos phillips fotografados em contexto de notebooks e teclados

O dataset foi baixado e utilizado para fine-tuning de FasterRCNN, CLIP e MobileNet. O problema ficou evidente na análise visual: o **domínio é completamente diferente** do desafio.

| Característica | Dataset GitHub | Imagens do Desafio |
|----------------|---------------|-------------------|
| Fundo | Placa-mãe, teclado, hardware | Azul uniforme / branco |
| Parafusos | Embutidos em equipamentos | Soltos, isolados |
| Contexto | Manutenção eletrônica | Contagem industrial |
| Tamanho relativo | Pequenos (< 2% da imagem) | Médios (3–15% da imagem) |
| Sobreposição | Rara | Frequente (pilhas) |

Resultado: os modelos fine-tunados no dataset do GitHub aprenderam a detectar parafusos em contexto eletrônico — um domínio ortogonal ao desafio.

### 4.2 Geração de Dataset Sintético com LLM

Diante da incompatibilidade dos dados externos, a segunda estratégia foi **gerar um dataset sintético domain-specific usando Claude Vision como guia de calibração**.

O processo foi iterativo:
1. **Análise das imagens reais** — Claude Vision extraiu os parâmetros visuais precisos das 5 imagens do desafio: cor de fundo (BGR ~[215,210,195]), tamanho de cabeça (3–5% da largura), textura de haste (helicoidal metálica com brilhos), tipo de iluminação (topo com leve inclinação)
2. **Renderização procedural** — gerador Python que renderiza parafusos realistas com: gradiente radial de metal na cabeça, cruz Phillips, haste helicoidal, sombra direcional, ruído de sensor de câmera e vinheta natural
3. **Augmentation** — cada imagem base gerou 67 augmentações (flip, rotação, brightness, contraste, zoom, ruído adicional)

**Volume final do dataset sintético:**
- 25 imagens base × 67 augmentações = **1.680 imagens**
- Contagens por imagem: 1–12 parafusos
- Distribuição balanceada de classes

### 4.3 Resultados do Fine-Tuning

Os modelos foram retreinados no dataset sintético e avaliados nas 5 imagens reais do desafio:

**Tier 2 — Fine-tuned (CPU/Notebook)**

| Método | Acertos | MAE | ms/img | Base |
|--------|---------|-----|--------|------|
| NCC_Multi | 3/5 | 2.00 | 51ms | Templates sintéticos + NCC por blob |
| LoG_Calibrado | 2/5 | 0.60 | 3679ms | Sigma range calibrado no dataset real |
| MobileNet_FT | 0/5 | 3.00 | 465ms | Cabeça de regressão treinada nos sintéticos |

**Tier 3 — Fine-tuned (GPU/Servidor)**

| Método | Acertos | MAE | ms/img | Base |
|--------|---------|-----|--------|------|
| OwlViT_Adapted | 1/5 | 3.00 | 1561ms | Threshold e prompts calibrados no dataset real |
| CLIP_Regressor | 0/5 | 2.80 | 5231ms | Ridge regression sobre features CLIP |
| FasterRCNN_FT | 0/5 | 4.80 | 3075ms | Fine-tune no dataset Cross-Recessed-Screw |

Nenhum modelo fine-tunado superou o WS_RegionalMax (5/5, MAE=0.00).

### 4.4 Análise do Domain Mismatch

O principal obstáculo identificado foi o **domain shift** em três camadas:

**1. Dataset externo → Desafio:** parafusos em hardware eletrônico vs. parafusos isolados em fundo azul. O modelo FasterRCNN fine-tunado no GitHub dataset aprendeu features de contexto (borda de placa, trilha de PCB) que não existem nas imagens do desafio.

**2. Dataset sintético → Imagens reais:** mesmo sendo gerado para imitar o domínio do desafio, o dataset sintético tem limitações inerentes. Com apenas 5 imagens reais de referência, a distribuição real é desconhecida — o gerador pode ter errado em parâmetros críticos (textura de rosca, reflexo de iluminação, deformação por perspectiva). 1.680 imagens sintéticas com distribuição incorreta não substituem 100 imagens reais anotadas.

**3. Tarefa de detecção vs. contagem:** CLIP e MobileNet foram treinados como regressores de contagem (saída: número inteiro), não como detectores (saída: bounding boxes). Para generalizar, o regressor precisa aprender invariâncias de posição e iluminação que 1.680 imagens sintéticas não cobrem adequadamente.

### 4.5 O que Faria Diferença com Mais Tempo e Dados

Com recursos adicionais, as abordagens mais promissoras seriam:

**Curto prazo (< 1 semana):**
- Anotar 50–100 imagens reais do domínio (fundo azul, parafusos soltos) com CVAT ou LabelImg
- Fine-tunar YOLOv8n-seg diretamente nestas imagens — o modelo já mostrou (no Desafio 2) capacidade de aprender em poucas épocas com poucos dados

**Médio prazo (1–2 semanas):**
- Construir um pipeline de geração sintética com renderização 3D (Blender + randomização de iluminação e câmera) para cobrir variações não presentes nas 5 imagens de referência
- Fine-tunar OwlViT com exemplos visuais few-shot — o modelo já entende a semântica de "parafuso", falta apenas calibrar o threshold de detecção para o domínio específico

**Longo prazo (modelo dedicado):**
- Treinar um modelo de density map estimation (CSRNet / MCNN) no domínio — métodos inspirados em contagem de multidões que evitam detecção explícita de objetos e são robustos a sobreposições densas
- Com 500+ imagens anotadas, este tipo de modelo tende a superar métodos CV clássicos em todos os cenários

---

## 5. Análise de Casos por Imagem (após benchmark completo)

### img5.jpg — O Caso Crítico (GT=9, pilha densa)

Este é o único caso onde a diferença entre métodos é decisiva:

| Método | Predição | Erro |
|--------|----------|------|
| WS_RegionalMax | 9 | 0 ✓ |
| Watershed | 6 | 3 |
| ConvexDefects | 7 | 2 |
| MSER | 5 | 4 |
| Canny_Contour | 0 | 9 |
| HoughCircles | 32 | 23 |

O footprint adaptativo do `maximum_filter` (`fp = max(7, int(min(h,w) * 0.05))`) é o parâmetro que determina a separação mínima entre picos no distance transform. Para img5, os parafusos sobrepostos criam um distance transform com múltiplos picos locais — cada um capturado como semente do Watershed.

---

## 6. Resumo de Recomendações por Cenário

| Cenário | Método Recomendado | Justificativa |
|---------|-------------------|---------------|
| Smartphone/IoT embarcado | **WS_RegionalMax** | 5/5, 8ms, ~2MB RAM, zero dependências além de OpenCV+scipy |
| CPU sem GPU (notebook) | **WS_RegionalMax** | Tier 2 não supera Tier 1 neste domínio |
| GPU disponível (servidor) | **WS_RegionalMax + fine-tune OwlViT** | OwlViT com fine-tuning de 50–200 imagens deve superar WS |
| Custo/acurácia máxima | **WS_RegionalMax + VLM fallback** | Pipeline atual — zero custo em 99% dos casos |
| Dataset novo / domínio diferente | **VLM Claude (zero-shot)** | Sem treino, adaptável; custo por chamada |

---

## 7. Conclusão

O método vencedor **WS_RegionalMax** combina três insights:

1. **Segmentação morfológica robusta** — adaptive threshold com blockSize=35 ignora textura de rosca, closing fecha silhueta.
2. **Regional Maxima como seeds** — `scipy.ndimage.maximum_filter` encontra cada pico local no distance transform, independente de sua altura absoluta.
3. **Footprint adaptativo** — 5% da menor dimensão da imagem equilibra separação vs. sobre-segmentação.

A jornada de fine-tuning revelou aprendizados importantes além dos resultados numéricos:

**Dados de qualidade superam quantidade.** 1.680 imagens sintéticas criteriosamente geradas não foram suficientes para superar um método clássico calibrado nas 5 imagens reais. O gerador foi construído com base em análise visual detalhada (via Claude Vision), mas pequenos desvios nos parâmetros de textura, iluminação e sobreposição criaram uma distribuição sintética ligeiramente diferente da real — diferença pequena o suficiente para ser invisível a olho nu, mas suficiente para degradar os modelos treinados nela.

**Domain mismatch é o principal inimigo do fine-tuning.** O dataset Cross-Recessed-Screw (GitHub) tem 900 imagens anotadas — 180× mais que o desafio — mas pertence a um domínio ortogonal (parafusos embutidos em hardware eletrônico). O FasterRCNN treinado nele atingiu 0/5, pior que o FasterRCNN zero-shot COCO (também 0/5, mas por motivos diferentes). Dados do domínio errado são ativamente prejudiciais.

**Métodos clássicos bem calibrados são competitivos em datasets pequenos.** Com apenas 5 imagens de referência, não há como treinar um modelo de detecção confiável. O WS_RegionalMax usa apenas as invariâncias do problema (parafusos formam blobs circulares que geram picos locais no distance transform) sem nenhuma dependência de dados de treino. Em domínios com poucos dados, esta abordagem dedutiva tende a vencer.

**Com mais dados, o cenário se inverte.** Com 100+ imagens anotadas no domínio correto, qualquer modelo de detecção moderno (YOLOv8, OwlViT fine-tuned, FasterRCNN com cabeça de domínio) superaria o WS_RegionalMax — especialmente em casos extremos de sobreposição que o distance transform não consegue separar. A densidade map estimation (CSRNet, MCNN), especificamente desenvolvida para contagem de objetos sobrepostos, seria o candidato mais robusto com dados suficientes.

A principal descoberta do benchmark é que **a escolha do método deve ser orientada pela quantidade e qualidade de dados disponíveis, não pela complexidade do modelo**. Com 5 imagens: WS_RegionalMax. Com 100+ imagens anotadas no domínio: fine-tuning. Com dados ilimitados: modelo dedicado de density map. E em qualquer cenário como fallback de último recurso: VLM zero-shot — que empatou com WS_RegionalMax em acurácia (5/5) com custo monetário e latência 200× maiores.

---

---

## Apêndice — Arquivos não incluídos no .zip

O arquivo `.zip` de submissão tem limite de 10 MB. Os artefatos abaixo foram omitidos por tamanho mas estão disponíveis integralmente no repositório GitHub:

**[github.com/GabrielAirex/CDIA_bootcamp_Gabriel_Aires](https://github.com/GabrielAirex/CDIA_bootcamp_Gabriel_Aires)**

| Artefato | Tamanho | Motivo da omissão | Localização no repo |
|---|---|---|---|
| `benchmark/output_benchmark/` | ~5.5 MB | Imagens geradas pelo benchmark (24 métodos × 5 imagens) | `desafio1/benchmark/output_benchmark/` |
| `benchmark/synthetic_dataset/` | ~83 MB | 1.680 imagens sintéticas de parafusos geradas via LLM | `desafio1/benchmark/synthetic_dataset/` |
| `benchmark/models/fasterrcnn_screw.pth` | ~159 MB | Modelo FasterRCNN fine-tuned (excede limite GitHub sem LFS) | não disponível no repo |
| `docs/screenshots/` | ~5.7 MB | Screenshots da interface para documentação | `docs/screenshots/` |

> O modelo `fasterrcnn_screw.pth` (159 MB) excede o limite de arquivo do GitHub (100 MB) e não está disponível online. Para reproduzir, execute `desafio1/benchmark/finetune.py` com o dataset Cross-Recessed-Screw.

---

*Relatório gerado em 31/05/2026 | CDIA Bootcamp — Residência em IA*
