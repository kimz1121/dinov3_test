"""Contrastive Disentanglement Model 구조 다이어그램 생성."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("figures/model_architecture.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── 캔버스 ─────────────────────────────────────────────────────────────────────
W, H = 14, 21
fig, ax = plt.subplots(figsize=(12, 18))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

# ── 색 팔레트 ──────────────────────────────────────────────────────────────────
C = dict(
    input_bg ='#E8F4FD', input_ec ='#4A90D9',
    pool_bg  ='#E8F8EE', pool_ec  ='#16A34A',
    cattn_bg ='#D1FAE5', cattn_ec ='#059669',
    tok_bg   ='#A7F3D0', tok_ec   ='#6EE7B7',
    q_bg     =['#6EE7B7','#34D399','#10B981','#059669'],
    shared_bg='#FFFBEB', shared_ec='#D97706',
    task_bg  ='#EBF5FF', task_ec  ='#2563EB',
    nuis_bg  ='#FFF3E0', nuis_ec  ='#EA580C',
    loss_bg  ='#F5F5F5', loss_ec  ='#6B7280',
    total_bg ='#F0F0F0', total_ec ='#374151',
)

# ── 헬퍼 ───────────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, bg, ec, lw=1.8, r=0.22):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=bg, ec=ec, lw=lw, zorder=3, clip_on=False))

def t(x, y, s, fs=10, fw='normal', col='#111827',
      ha='center', va='center', style='normal'):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col,
            ha=ha, va=va, style=style, zorder=5, clip_on=False)

def arr(x1, y1, x2, y2, col='#374151', lw=1.8,
        conn='arc3,rad=0', ls='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=4,
                annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                linestyle=ls, connectionstyle=conn))


# ══════════════════════════════════════════════════════════════════════════════
# 제목
# ══════════════════════════════════════════════════════════════════════════════
t(7, 20.5, 'Contrastive Disentanglement Model', fs=15, fw='bold')
t(7, 20.0, 'DINOv3 patch tokens → AttentionPool → Task / Nuisance embedding',
  fs=9.5, col='#6B7280', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# 1. 입력
# ══════════════════════════════════════════════════════════════════════════════
box(7, 19.15, 13.0, 0.9, C['input_bg'], C['input_ec'])
t(7, 19.43, 'Input: Pre-computed DINOv3 Patch Tokens', fs=11, fw='bold', col='#1E40AF')
t(7, 18.98, 'shape  (B, N, 384)      N = n_frames × H_patches × W_patches',
  fs=9.5, col='#374151', style='italic')

arr(7, 18.70, 7, 17.88)

# ══════════════════════════════════════════════════════════════════════════════
# 2. AttentionPool 외부 박스
# ══════════════════════════════════════════════════════════════════════════════
# 외부 박스: cy=15.85, h=4.05 → top=17.875, bot=13.825
box(7, 15.85, 13.0, 4.05, C['pool_bg'], C['pool_ec'], lw=2.2, r=0.3)
t(7, 17.68, 'AttentionPool', fs=12, fw='bold', col='#065F46')

# ── 2-a. 패치 토큰 시퀀스 (Keys · Values) ──────────────────────────────────────
t(7, 17.32, 'Patch Token Sequence  (Keys  ·  Values)', fs=9.5, col='#065F46', fw='bold')

# 토큰 박스 8개 (균등 배치, 박스 안에 완전히 들어오도록)
tok_xs = [1.5, 2.6, 3.7, 4.8, 5.9, 7.0, 8.1, 9.2]
for i, tx in enumerate(tok_xs):
    box(tx, 16.85, 0.82, 0.46, C['tok_bg'], C['tok_ec'], lw=0.9, r=0.07)
    t(tx, 16.85, f't{i+1}', fs=8, col='#064E3B', fw='bold')
# 말줄임
t(10.1, 16.85, '···', fs=12, col='#065F46')
box(11.0, 16.85, 0.82, 0.46, C['tok_bg'], C['tok_ec'], lw=0.9, r=0.07)
t(11.0, 16.85, 'tN', fs=8, col='#064E3B', fw='bold')

# ── 2-b. 토큰 → Cross-Attention 화살표 (K, V) ──────────────────────────────────
# 여러 토큰에서 내려오는 화살표 3개로 표현
for ax_x in [3.7, 6.4, 9.2]:
    arr(ax_x, 16.62, ax_x, 16.22, col='#059669', lw=1.3)
t(7, 16.43, '(K,  V)', fs=9, col='#059669', fw='bold')

# ── 2-c. Cross-Attention 내부 박스 ────────────────────────────────────────────
# cx=5.5 → 쿼리 박스가 오른쪽에 들어올 공간 확보
box(5.5, 15.73, 5.5, 0.82, C['cattn_bg'], C['cattn_ec'], lw=1.8, r=0.18)
t(5.5, 15.93, 'Cross-Attention', fs=11, fw='bold', col='#065F46')
t(5.5, 15.57, 'Q  ×  softmax(QKᵀ/√d)  ×  V', fs=8.5, col='#374151')

# ── 2-d. K Learnable Queries — 오른쪽에서 화살표로 입력 ────────────────────────
# 쿼리 레이블 (위)
t(10.65, 16.32, 'K Learnable Queries  (Q)', fs=9.5, col='#065F46', fw='bold')

# 쿼리 벡터 박스 4개
q_xs = [8.85, 9.65, 10.45, 11.25]
for i, (qx, fc) in enumerate(zip(q_xs, C['q_bg'])):
    box(qx, 15.73, 0.65, 0.48, fc, '#065F46', lw=1.0, r=0.08)
    t(qx, 15.73, f'q{i+1}', fs=9, fw='bold', col='#064E3B')
t(10.05, 15.38, '(learnable parameters)', fs=8, col='#065F46', style='italic')

# 쿼리 → Cross-Attention 화살표 (왼쪽 방향)
arr(8.55, 15.73, 8.25, 15.73, col='#065F46', lw=2.2)

# ── 2-e. Cross-Attention 출력 → 아래 ──────────────────────────────────────────
arr(5.5, 15.32, 7.0, 14.62, col='#059669', lw=1.5, conn='arc3,rad=-0.15')
t(7.0, 14.38, 'output  (B, K, 384)  →  flatten  →  (B, K × 384)',
  fs=9, col='#374151', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# 3. 공유 표현
# ══════════════════════════════════════════════════════════════════════════════
arr(7, 13.83, 7, 13.18)

box(7, 12.82, 6.5, 0.62, C['shared_bg'], C['shared_ec'], lw=1.8)
t(7, 13.01, 'Shared representation  (B, K × 384)', fs=10.5, fw='bold', col='#92400E')
t(7, 12.65, 'same trunk  →  feeds both task & nuisance heads',
  fs=8.5, col='#6B7280', style='italic')

arr(7, 12.51, 3.2, 11.70, col=C['task_ec'], lw=1.8)
arr(7, 12.51, 10.8, 11.70, col=C['nuis_ec'], lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Projection Heads
# ══════════════════════════════════════════════════════════════════════════════
box(3.2, 11.08, 5.8, 1.15, C['task_bg'], C['task_ec'], lw=2.0)
t(3.2, 11.50, 'Task Projection Head', fs=11, fw='bold', col='#1E3A8A')
t(3.2, 11.15, 'Linear → GELU → Linear', fs=9, col='#334155')
t(3.2, 10.80, '(K×384 → 512 → 128)  +  LayerNorm  +  L2-norm', fs=8.5, col='#334155')

box(10.8, 11.08, 5.8, 1.15, C['nuis_bg'], C['nuis_ec'], lw=2.0)
t(10.8, 11.50, 'Nuisance Projection Head', fs=11, fw='bold', col='#7C2D12')
t(10.8, 11.15, 'Linear → GELU → Linear', fs=9, col='#334155')
t(10.8, 10.80, '(K×384 → 512 → 64)   +  LayerNorm  +  L2-norm', fs=8.5, col='#334155')

arr(3.2, 10.50, 3.2, 9.78, col=C['task_ec'])
arr(10.8, 10.50, 10.8, 9.78, col=C['nuis_ec'])

# ══════════════════════════════════════════════════════════════════════════════
# 5. Embedding 공간
# ══════════════════════════════════════════════════════════════════════════════
box(3.2, 9.38, 5.2, 0.72, C['task_bg'], C['task_ec'], lw=2.5)
t(3.2, 9.58, 'z_task  ∈  ℝ¹²⁸', fs=12, fw='bold', col=C['task_ec'])
t(3.2, 9.21, 'task-discriminative embedding', fs=9, col='#374151')

box(10.8, 9.38, 5.2, 0.72, C['nuis_bg'], C['nuis_ec'], lw=2.5)
t(10.8, 9.58, 'z_nuis  ∈  ℝ⁶⁴', fs=12, fw='bold', col=C['nuis_ec'])
t(10.8, 9.21, 'nuisance (camera) embedding', fs=9, col='#374151')

# ══════════════════════════════════════════════════════════════════════════════
# 6. 손실 함수
# ══════════════════════════════════════════════════════════════════════════════
arr(3.2, 9.02, 2.2, 8.22, col=C['task_ec'])
arr(10.8, 9.02, 11.8, 8.22, col=C['nuis_ec'])
ax.annotate('', xy=(7, 8.22), xytext=(3.2, 9.02),
            arrowprops=dict(arrowstyle='->', color=C['loss_ec'], lw=1.5,
                            connectionstyle='arc3,rad=-0.22'), zorder=4)
ax.annotate('', xy=(7, 8.22), xytext=(10.8, 9.02),
            arrowprops=dict(arrowstyle='->', color=C['loss_ec'], lw=1.5,
                            connectionstyle='arc3,rad=0.22'), zorder=4)

box(2.2, 7.82, 3.8, 0.72, C['loss_bg'], C['task_ec'], lw=1.8)
t(2.2, 8.02, 'ℒ_task  (SupCon)', fs=11, fw='bold', col=C['task_ec'])
t(2.2, 7.66, 'same task_id  →  attract', fs=9, col='#374151')

box(7, 7.82, 3.5, 0.72, C['loss_bg'], C['loss_ec'], lw=1.8)
t(7, 8.04, 'ℒ_ortho', fs=11, fw='bold', col='#374151')
t(7, 7.68, 'z_task ⊥ z_nuis', fs=9, col='#374151')

box(11.8, 7.82, 3.8, 0.72, C['loss_bg'], C['nuis_ec'], lw=1.8)
t(11.8, 8.02, 'ℒ_nuis  (SupCon)', fs=11, fw='bold', col=C['nuis_ec'])
t(11.8, 7.66, 'same camera_id  →  attract', fs=9, col='#374151')

# ══════════════════════════════════════════════════════════════════════════════
# 7. 전체 손실
# ══════════════════════════════════════════════════════════════════════════════
arr(2.2, 7.46, 4.0, 6.82, col=C['task_ec'], lw=1.5)
arr(7,   7.46, 7,   6.82, col=C['loss_ec'], lw=1.5)
arr(11.8, 7.46, 10.0, 6.82, col=C['nuis_ec'], lw=1.5)

box(7, 6.45, 11.0, 0.68, C['total_bg'], C['total_ec'], lw=2.0, r=0.2)
t(7, 6.65, 'ℒ  =  ℒ_task  +  λ_n · ℒ_nuis  +  λ_o · ℒ_ortho',
  fs=12, fw='bold', col='#111827')
t(7, 6.28, 'λ_n = 1.0      λ_o = 0.005',
  fs=9, col='#6B7280', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# 8. 지도 레이블
# ══════════════════════════════════════════════════════════════════════════════
box(2.2, 5.28, 3.2, 0.62, C['task_bg'], C['task_ec'], lw=1.4, r=0.12)
t(2.2, 5.48, 'task label', fs=10, fw='bold', col=C['task_ec'])
t(2.2, 5.11, '(e.g. "OpenDrawer")', fs=8.5, col='#6B7280')

box(11.8, 5.28, 3.2, 0.62, C['nuis_bg'], C['nuis_ec'], lw=1.4, r=0.12)
t(11.8, 5.48, 'camera label', fs=10, fw='bold', col=C['nuis_ec'])
t(11.8, 5.11, '(e.g. "eye_in_hand")', fs=8.5, col='#6B7280')

arr(2.2, 5.59, 2.2, 7.46, col=C['task_ec'], lw=1.2, ls='dashed')
arr(11.8, 5.59, 11.8, 7.46, col=C['nuis_ec'], lw=1.2, ls='dashed')

# ══════════════════════════════════════════════════════════════════════════════
# 9. Footer
# ══════════════════════════════════════════════════════════════════════════════
t(7, 0.5,
  'K=4 (default)  ·  d_task=128  ·  d_nuis=64  ·  hidden=512  ·  heads=8  ·  temperature=0.1',
  fs=9, col='#9CA3AF')

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f"saved → {OUT}")
