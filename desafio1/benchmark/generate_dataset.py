"""
Gerador v2 — calibrado por análise profunda via Claude Vision.

Parâmetros reais extraídos das 5 imagens:
  Fundo:   azul-claro/branco-creme BGR~[215,210,195], textura câmera muito fina
  Cabeça:  BGR[165,160,155], highlight BGR[220,218,215], sombra BGR[95,95,100]
  Haste:   BGR[140,138,135], rosca helicoidal metálica com brilhos finos
  Tamanho: cabeça 3-5% da largura, haste 8-15% da altura
  Sombra:  direcional suave (não circular/dura)
  View:    topo com leve inclinação, orientação aleatória

Geração:
  - N_BASE imagens base renderizadas
  - N_AUGS augmentations por base
"""

import json, math, random
import cv2
import numpy as np
from pathlib import Path

OUT_DIR  = Path(__file__).parent / "synthetic_dataset"
OUT_DIR.mkdir(exist_ok=True)

IMG_W, IMG_H = 640, 640
RNG = random.Random(42)
NP_RNG = np.random.RandomState(42)


# ─────────────────────────────────────────────────────────────────────────────
# Background realista
# ─────────────────────────────────────────────────────────────────────────────

def make_background(rng: random.Random, np_rng: np.random.RandomState) -> np.ndarray:
    """
    Fundo azul-claro/branco-creme com textura de sensor de câmera (não Perlin).
    Variação entre imagens: algumas mais azuladas, outras mais brancas/creme.
    """
    style = rng.choice(["blue_light", "white_cream", "blue_pale"])

    if style == "blue_light":
        B = rng.randint(208, 225)
        G = rng.randint(200, 215)
        R = rng.randint(185, 200)
    elif style == "white_cream":
        base = rng.randint(215, 235)
        B = base - rng.randint(0, 8)
        G = base - rng.randint(3, 10)
        R = base - rng.randint(10, 18)
    else:  # blue_pale
        B = rng.randint(218, 235)
        G = rng.randint(210, 225)
        R = rng.randint(195, 210)

    bg = np.full((IMG_H, IMG_W, 3), [B, G, R], dtype=np.float32)

    # Gradiente suave de iluminação (vinheta natural de câmera celular)
    Y, X = np.mgrid[:IMG_H, :IMG_W]
    cx, cy = IMG_W / 2 + rng.uniform(-80, 80), IMG_H / 2 + rng.uniform(-80, 80)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    vignette = 1.0 - 0.10 * np.clip(dist / (IMG_W * 0.65) - 0.3, 0, 1)
    for c in range(3):
        bg[:, :, c] *= vignette

    # Ruído muito fino de sensor (não Perlin — apenas random pixel noise)
    sensor_noise = np_rng.randn(IMG_H, IMG_W) * 2.5
    for c in range(3):
        bg[:, :, c] += sensor_noise

    return np.clip(bg, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Renderização de parafuso realista
# ─────────────────────────────────────────────────────────────────────────────

def render_screw(canvas: np.ndarray, cx: int, cy: int,
                 head_r: int, angle_deg: float,
                 rng: random.Random, np_rng: np.random.RandomState) -> None:
    """
    Renderiza hex bolt metálico com parâmetros calibrados.
    cx, cy: centro da CABEÇA.
    head_r: raio da cabeça em pixels.
    angle_deg: orientação do eixo da haste (0=para baixo, 90=para direita).
    """
    # ── Cores calibradas ──────────────────────────────────────────────────────
    head_mid  = np.array([165, 160, 155], np.float32)
    head_hi   = np.array([220, 218, 215], np.float32)
    head_shad = np.array([100, 98,  95],  np.float32)
    shaft_col = np.array([140, 138, 135], np.float32)
    shadow_c  = np.array([95,  95,  100], np.float32)

    # Variação individual leve
    var = np_rng.randint(-8, 8, 3).astype(np.float32)
    head_mid  = np.clip(head_mid  + var, 80, 240)
    head_hi   = np.clip(head_hi   + var * 0.5, 150, 250)
    shaft_col = np.clip(shaft_col + var, 80, 200)

    # ── Parâmetros geométricos ────────────────────────────────────────────────
    shaft_w   = max(3, int(head_r * 0.55))       # largura da haste
    shaft_len = int(head_r * rng.uniform(2.2, 3.5))  # comprimento da haste

    angle_rad = math.radians(angle_deg)
    ax = math.cos(angle_rad)   # direção longitudinal
    ay = math.sin(angle_rad)
    px_ = -ay                  # perpendicular
    py_ = ax

    # ── Sombra direcional suave ───────────────────────────────────────────────
    # Fonte de luz: superior-esquerda; sombra para baixo-direita
    shad_dx = int(head_r * 0.35)
    shad_dy = int(head_r * 0.45)

    # Sombra da haste
    shad_layer = np.zeros((IMG_H, IMG_W), np.float32)
    for t in range(shaft_len + head_r):
        tx = int((cx + shad_dx) + ax * (head_r * 0.3 + t))
        ty = int((cy + shad_dy) + ay * (head_r * 0.3 + t))
        fade = max(0.0, 1.0 - t / (shaft_len + head_r))
        for s in range(-shaft_w - 2, shaft_w + 3):
            sx = int(tx + px_ * s)
            sy = int(ty + py_ * s)
            if 0 <= sy < IMG_H and 0 <= sx < IMG_W:
                lat = abs(s) / (shaft_w + 2.0)
                shad_layer[sy, sx] = max(shad_layer[sy, sx],
                                         0.35 * fade * (1 - lat))

    # Sombra da cabeça (elipse suave)
    shad_mask = np.zeros((IMG_H, IMG_W), np.float32)
    cv2.ellipse(shad_mask, (cx + shad_dx, cy + shad_dy),
                (head_r + 3, int(head_r * 0.8)), int(angle_deg),
                0, 360, 1.0, -1)
    shad_layer = np.maximum(shad_layer, shad_mask * 0.35)

    # Blur suave na sombra
    shad_layer = cv2.GaussianBlur(shad_layer, (int(head_r * 0.8) | 1, int(head_r * 0.8) | 1), head_r * 0.3)

    # Aplica sombra no canvas
    for c in range(3):
        col_s = shadow_c[c] * 0.6
        canvas[:, :, c] = np.clip(
            canvas[:, :, c].astype(np.float32) * (1 - shad_layer * 0.55)
            + col_s * shad_layer * 0.3,
            0, 255
        ).astype(np.uint8)

    # ── Haste com rosca helicoidal ────────────────────────────────────────────
    # Rosca real: cristas finas com reflexo, vales escuros — não grade retangular
    thread_pitch = max(3, int(head_r * 0.35))  # passo da rosca

    for t in range(shaft_len):
        frac_len = t / max(shaft_len - 1, 1)
        tx = int(cx + ax * (head_r * 0.25 + t))
        ty = int(cy + ay * (head_r * 0.25 + t))

        if not (0 <= ty < IMG_H and 0 <= tx < IMG_W):
            continue

        # Helicoide: posição angular da crista varia com t
        helix_phase = (t % thread_pitch) / thread_pitch  # 0..1
        # Crista no top-left para cada passo
        crest_lat = (helix_phase - 0.5) * shaft_w  # onde está a crista

        # Escurecimento progressivo ao longo da haste
        dim = 1.0 - 0.30 * frac_len

        for s in range(-shaft_w // 2, shaft_w // 2 + 1):
            sx = int(tx + px_ * s)
            sy = int(ty + py_ * s)
            if not (0 <= sy < IMG_H and 0 <= sx < IMG_W):
                continue

            # Gradiente lateral (arestas mais escuras)
            lat_n = abs(s) / (shaft_w / 2 + 0.5)
            edge_dim = 1.0 - 0.40 * lat_n

            # Rosca helicoidal: brilho varia com distância à crista
            dist_to_crest = abs(s - crest_lat)
            if dist_to_crest < 1.2:
                thread_factor = 1.15  # crista brilhante
            elif dist_to_crest < 2.5:
                thread_factor = 0.85  # vale escuro
            else:
                thread_factor = 1.00  # corpo normal

            pixel = (shaft_col * dim * edge_dim * thread_factor).astype(np.uint8)
            canvas[sy, sx] = np.clip(pixel, 0, 230)

    # ── Cabeça hexagonal ──────────────────────────────────────────────────────
    hex_pts = np.array([
        [int(cx + head_r * math.cos(math.radians(60 * i + angle_deg + 30))),
         int(cy + head_r * math.sin(math.radians(60 * i + angle_deg + 30)))]
        for i in range(6)
    ], dtype=np.int32)

    # Preenchimento base
    cv2.fillPoly(canvas, [hex_pts], tuple(head_mid.astype(int).tolist()))

    # Gradiente metallic: mais brilhante no topo-esquerdo (luz superior-esquerda)
    y0 = max(0,  cy - head_r - 2)
    y1 = min(IMG_H, cy + head_r + 2)
    x0 = max(0,  cx - head_r - 2)
    x1 = min(IMG_W, cx + head_r + 2)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if cv2.pointPolygonTest(hex_pts, (float(x), float(y)), False) < 0:
                continue
            # Direção luz: superior-esquerda
            nx = (x - cx) / (head_r + 0.1)
            ny = (y - cy) / (head_r + 0.1)
            light = -0.55 * nx - 0.70 * ny  # dot com (−0.55, −0.70)
            t = max(0.0, min(1.0, 0.5 + light * 0.7))
            # Especular perto do ponto de luz
            d_norm = math.hypot(nx, ny)
            spec = max(0.0, (1.0 - d_norm * 3.5 + 2.0 * t)) if t > 0.55 else 0.0
            spec = min(spec, 0.7)
            color = head_mid * (1 - t) + head_hi * t + np.array([255, 255, 255]) * spec
            canvas[y, x] = np.clip(color, 0, 255).astype(np.uint8)

    # Bordas hexagonais (arestas com sombra + reflexo direcional)
    for i in range(6):
        p1 = hex_pts[i]
        p2 = hex_pts[(i + 1) % 6]
        # Aresta na direção da luz = reflexo claro; oposta = sombra
        mid_nx = (p1[0] + p2[0]) / 2 - cx
        mid_ny = (p1[1] + p2[1]) / 2 - cy
        edge_light = -0.55 * mid_nx - 0.70 * mid_ny
        if edge_light > 0:
            edge_color = tuple(np.clip(head_hi * 0.9, 0, 255).astype(int).tolist())
        else:
            edge_color = tuple(head_shad.astype(int).tolist())
        cv2.line(canvas, tuple(p1), tuple(p2), edge_color, 1)

    # Furo central (recess de chave Allen / identificador do parafuso)
    inner_r = max(2, int(head_r * 0.22))
    cv2.circle(canvas, (cx, cy), inner_r + 1,
               tuple(head_shad.astype(int).tolist()), -1)
    cv2.circle(canvas, (cx, cy), inner_r,
               tuple((head_shad * 0.7).astype(int).tolist()), -1)
    # Mini highlight no furo
    cv2.circle(canvas, (cx - inner_r//3, cy - inner_r//3), max(1, inner_r//3),
               tuple(head_hi.astype(int).tolist()), -1)


# ─────────────────────────────────────────────────────────────────────────────
# Gera uma cena com N parafusos
# ─────────────────────────────────────────────────────────────────────────────

def generate_scene(n_screws: int, seed: int) -> tuple[np.ndarray, int, list]:
    """
    Retorna (canvas, n_placed, boxes).
    boxes: lista de [x1, y1, x2, y2] — bbox de cada parafuso (inclui haste).
    """
    rng     = random.Random(seed)
    np_rng  = np.random.RandomState(seed)
    canvas  = make_background(rng, np_rng)

    min_head_r = int(IMG_W * 0.015)
    max_head_r = int(IMG_W * 0.025)

    placed = []
    boxes  = []
    for _ in range(n_screws * 8):
        if len(placed) >= n_screws:
            break
        head_r    = rng.randint(min_head_r, max_head_r)
        angle     = rng.uniform(0, 360)
        shaft_len = int(head_r * rng.uniform(2.4, 3.5))

        margin = head_r + 4
        cx = rng.randint(margin, IMG_W - margin)
        cy = rng.randint(margin, IMG_H - margin)

        ok = True
        for ox, oy, or_ in placed:
            if math.hypot(cx - ox, cy - oy) < (head_r + or_) * 0.75:
                ok = False
                break
        if ok:
            render_screw(canvas, cx, cy, head_r, angle, rng, np_rng)
            placed.append((cx, cy, head_r))

            # Bbox: engloba cabeça + haste
            ax = math.cos(math.radians(angle))
            ay = math.sin(math.radians(angle))
            tip_x = cx + ax * (head_r * 0.25 + shaft_len)
            tip_y = cy + ay * (head_r * 0.25 + shaft_len)
            shaft_w = max(3, int(head_r * 0.55))
            all_x = [cx - head_r, cx + head_r, int(tip_x) - shaft_w, int(tip_x) + shaft_w]
            all_y = [cy - head_r, cy + head_r, int(tip_y) - shaft_w, int(tip_y) + shaft_w]
            x1 = max(0, min(all_x))
            y1 = max(0, min(all_y))
            x2 = min(IMG_W, max(all_x))
            y2 = min(IMG_H, max(all_y))
            boxes.append([x1, y1, x2, y2])

    return canvas, len(placed), boxes


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment(img: np.ndarray, count: int, aug_id: int, rng: random.Random) -> tuple:
    h, w = img.shape[:2]
    aug  = aug_id % 20

    # Flips
    if aug == 0: return cv2.flip(img, 1), count
    if aug == 1: return cv2.flip(img, 0), count
    if aug == 2: return cv2.flip(img, -1), count

    # Rotações exatas
    if aug in (3, 4, 5, 6):
        a = [90, 180, 270, rng.uniform(10, 50)][aug - 3]
        M = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), count

    # Brilho
    if aug == 7:
        f = rng.uniform(0.82, 0.95)
        return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8), count
    if aug == 8:
        f = rng.uniform(1.05, 1.18)
        return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8), count

    # Contraste
    if aug == 9:
        a = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8), count

    # Ruído sensor
    if aug == 10:
        n = np.random.randn(*img.shape) * rng.uniform(3, 8)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8), count

    # Blur câmera
    if aug == 11:
        k = rng.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0), count

    # Zoom in/out
    if aug == 12:
        s = rng.uniform(0.88, 1.12)
        new_sz = int(w * s)
        res = cv2.resize(img, (new_sz, new_sz))
        if s > 1:
            off = (new_sz - w) // 2
            return res[off:off+h, off:off+w], count
        else:
            pad = np.full((h, w, 3), 210, np.uint8)
            off = (w - new_sz) // 2
            pad[off:off+new_sz, off:off+new_sz] = res
            return pad, count

    # Jitter de cor de fundo (simula diferentes iluminações)
    if aug == 13:
        res = img.copy().astype(np.float32)
        shift_b = rng.uniform(-12, 12)
        shift_g = rng.uniform(-8, 8)
        shift_r = rng.uniform(-8, 8)
        # Aplica shift apenas nas regiões de fundo (pixels claros)
        bg_mask = (img.mean(axis=2) > 160).astype(np.float32)
        res[:,:,0] += shift_b * bg_mask
        res[:,:,1] += shift_g * bg_mask
        res[:,:,2] += shift_r * bg_mask
        return np.clip(res, 0, 255).astype(np.uint8), count

    # Sharpening
    if aug == 14:
        k = np.array([[-0.5,-0.5,-0.5],[-0.5,5,-0.5],[-0.5,-0.5,-0.5]])
        sharp = cv2.filter2D(img, -1, k)
        blend = cv2.addWeighted(img, 0.6, sharp, 0.4, 0)
        return blend, count

    # JPEG compression
    if aug == 15:
        q = rng.randint(65, 88)
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR), count

    # Flip + rotação aleatória
    if aug == 16:
        fl = cv2.flip(img, rng.choice([-1, 0, 1]))
        a  = rng.uniform(0, 360)
        M  = cv2.getRotationMatrix2D((w/2, h/2), a, 1.0)
        return cv2.warpAffine(fl, M, (w, h), borderMode=cv2.BORDER_REFLECT), count

    # Sombra aleatória no canto (simula objeto parcialmente fora do frame)
    if aug == 17:
        res = img.copy().astype(np.float32)
        side = rng.choice(["top", "bottom", "left", "right"])
        strength = rng.uniform(0.06, 0.15)
        size = rng.randint(h // 6, h // 3)
        if side == "top":    res[:size, :, :] *= (1 - strength)
        elif side == "bottom": res[-size:, :, :] *= (1 - strength)
        elif side == "left":   res[:, :size, :] *= (1 - strength)
        else:                  res[:, -size:, :] *= (1 - strength)
        return np.clip(res, 0, 255).astype(np.uint8), count

    # Rotação + escala combinadas
    if aug == 18:
        a = rng.uniform(0, 360)
        s = rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((w/2, h/2), a, s)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), count

    # Blur direcional (motion blur leve)
    if aug == 19:
        sz = rng.randint(3, 7)
        kernel = np.zeros((sz, sz))
        if rng.random() > 0.5:
            kernel[sz // 2, :] = 1.0 / sz
        else:
            for i in range(sz): kernel[i, i] = 1.0 / sz
        return cv2.filter2D(img, -1, kernel), count

    return img.copy(), count


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_base: int = 80, n_augs: int = 20):
    """
    Gera n_base cenas base + n_augs augmentations cada.
    Total: n_base * (1 + n_augs) imagens.
    Distribuição uniforme de contagens 1-12.
    """
    print(f"Gerando {n_base} cenas base + {n_augs} augs cada = {n_base*(1+n_augs)} imagens totais...")

    # Distribuição balanceada de contagens
    counts_pool = (list(range(1, 13)) * (n_base // 12 + 2))[:n_base]
    random.Random(7).shuffle(counts_pool)

    all_records = []

    for i, n_target in enumerate(counts_pool):
        img, actual_n, boxes = generate_scene(n_target, seed=i * 137 + 1)
        fname = f"base_{i:04d}_n{actual_n:02d}.jpg"
        cv2.imwrite(str(OUT_DIR / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec = {"file": fname, "count": actual_n, "boxes": boxes, "type": "base"}
        all_records.append(rec)

        rng_aug = random.Random(i * 999)
        for a in range(n_augs):
            aug_img, aug_n = augment(img, actual_n, a, rng_aug)
            aname = f"base_{i:04d}_n{actual_n:02d}_aug{a:02d}.jpg"
            cv2.imwrite(str(OUT_DIR / aname), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            all_records.append({"file": aname, "count": aug_n, "type": f"aug_{a}"})

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_base} cenas — última: {actual_n} parafusos")

    gt_path = OUT_DIR / "ground_truth.json"
    gt_path.write_text(json.dumps(all_records, indent=2))

    from collections import Counter
    dist = Counter(r["count"] for r in all_records)
    total = len(all_records)
    bases = sum(1 for r in all_records if r["type"] == "base")
    print(f"\n✓ Dataset gerado:")
    print(f"  Base:      {bases}")
    print(f"  Augs:      {total - bases}")
    print(f"  Total:     {total}")
    print(f"  Dist:      {dict(sorted(dist.items()))}")
    print(f"  GT:        {gt_path}")
    return all_records


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=int, default=80, help="cenas base")
    p.add_argument("--augs", type=int, default=20, help="augmentations por cena")
    p.add_argument("--preview", action="store_true", help="salva grid 3x3 em /tmp/")
    args = p.parse_args()

    # Limpa dataset anterior
    for f in OUT_DIR.glob("*.jpg"): f.unlink()

    records = generate_dataset(n_base=args.base, n_augs=args.augs)

    if args.preview:
        bases = [OUT_DIR / r["file"] for r in records if r["type"] == "base"][:9]
        grid, row = [], []
        for pb in bases:
            img = cv2.imread(str(pb))
            n   = pb.stem.split("_n")[1].split("_")[0]
            img = cv2.resize(img, (213, 213))
            cv2.putText(img, f"n={n}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.putText(img, f"n={n}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 255), 1)
            row.append(img)
            if len(row) == 3:
                grid.append(np.hstack(row))
                row = []
        if row:
            while len(row) < 3: row.append(np.zeros((213,213,3), np.uint8))
            grid.append(np.hstack(row))
        cv2.imwrite("/tmp/preview_v2.jpg", np.vstack(grid))
        print("Preview: /tmp/preview_v2.jpg")
