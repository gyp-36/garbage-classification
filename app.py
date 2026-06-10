"""
垃圾分类识别系统 - 极简高对比度常驻平铺版
"""
import gradio as gr
import torch
import json
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    from models.efficientnet import create_model
    from utils.transforms import get_val_transforms
except ImportError:
    def create_model(num_classes, pretrained): return torch.nn.Linear(10, num_classes)
    def get_val_transforms(): return lambda x: torch.zeros(3, 224, 224)

# 配置路径
MODEL_PATH = './outputs/models/best_model.pth'
CLASS_CONFIG = './data/garbage_21classes.json'

try:
    with open(CLASS_CONFIG) as f:
        CLASS_CONFIG_DATA = json.load(f)
except Exception:
    CLASS_CONFIG_DATA = {
        'classes': [
            {'name': '报纸', 'parent': '可回收物'},
            {'name': '塑料瓶', 'parent': '可回收物'},
            {'name': '易拉罐', 'parent': '可回收物'},
            {'name': '剩饭', 'parent': '厨余垃圾'},
            {'name': '果皮', 'parent': '厨余垃圾'},
            {'name': '菜叶', 'parent': '厨余垃圾'},
            {'name': '电池', 'parent': '有害垃圾'},
            {'name': '过期药品', 'parent': '有害垃圾'},
            {'name': '烟头', 'parent': '其他垃圾'},
            {'name': '污染纸张', 'parent': '其他垃圾'},
        ]
    }

CATEGORIES = {
    '可回收物': {'color': '#047857', 'bg': '#ECFDF5', 'emoji': '♻️', 'items': []},
    '厨余垃圾': {'color': '#B45309', 'bg': '#FFFBEB', 'emoji': '🍎', 'items': []},
    '有害垃圾': {'color': '#B91C1C', 'bg': '#FEF2F2', 'emoji': '☠️', 'items': []},
    '其他垃圾': {'color': '#111827', 'bg': '#F3F4F6', 'emoji': '🗑️', 'items': []},
}

for item in CLASS_CONFIG_DATA['classes']:
    parent = item['parent']
    if parent in CATEGORIES:
        CATEGORIES[parent]['items'].append(item['name'])

# 高对比度终极样式
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #FAFAF9;
    --surface: #FFFFFF;
    --text: #000000;             /* 全局纯黑 */
    --border: #71717A;           /* 加深灰色边框 */
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
    background: var(--bg);
}

/* 导航栏 */
.navbar { display: flex; justify-content: space-between; align-items: center; padding-bottom: 24px; border-bottom: 2px solid var(--border); margin-bottom: 32px; }
.navbar-brand { display: flex; align-items: center; gap: 12px; }
.navbar-logo { width: 40px; height: 40px; background: #000000; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
.navbar-title { font-size: 22px !important; font-weight: 800 !important; color: var(--text) !important; }

/* 面板外框 */
.product-panel {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}
.panel-title-bar {
    padding: 14px 20px;
    border-bottom: 2px solid var(--border);
    background: #E4E4E7;
}
.panel-title-text { font-size: 16px !important; font-weight: 800 !important; color: var(--text) !important; }

.product-panel .form, .product-panel .image-container, .product-panel .gr-box {
    border: none !important;
    background: transparent !important;
}

/* ⚡ 识别结果字体强制加粗、变漆黑 ⚡ */
.product-panel .label-container { padding: 16px !important; }
.product-panel .label-item { 
    color: #000000 !important; 
    font-weight: 900 !important; 
    font-size: 17px !important; 
}
.product-panel .confidence-text { 
    color: #000000 !important; 
    font-weight: 900 !important; 
    font-size: 16px !important;
}
/* 提升结果标签中非 Top-1 类别的次级文字对比度 */
.product-panel .shrunk-label, .product-panel .meta-text {
    color: #000000 !important;
    font-weight: 800 !important;
}
.product-panel .progress-bar { background-color: #000000 !important; }

/* 静态指南平铺面板 */
.guide-container {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
}
.guide-row {
    margin-bottom: 16px;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #000000 !important;
    line-height: 1.6 !important;
    border: 1px solid rgba(0, 0, 0, 0.15);
}
.guide-row:last-child { margin-bottom: 0; }
"""

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = create_model(num_classes=21, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device).eval()
    except Exception:
        model = torch.nn.Linear(10, 21).to(device).eval()
    return model, device

def predict(image, model, device):
    if image is None:
        return {}
    try:
        transform = get_val_transforms()
        img = Image.fromarray(image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)

        result = {}
        for i, item in enumerate(CLASS_CONFIG_DATA['classes']):
            if i < probs.shape[1]:
                result[item['name']] = float(probs[0][i])
        return result
    except Exception as e:
        return {"测试数据": 1.0}

def build_guide_html():
    """直接生成平铺介绍的 HTML"""
    html = '<div class="guide-container">'
    for cat_name, info in CATEGORIES.items():
        items_str = "、".join(info['items']) if info['items'] else "暂无数据"
        # 组装单行内容
        html += f"""
        <div class="guide-row" style="background-color: {info['bg']};">
            <span style="color: {info['color']}; font-weight: 900;">{info['emoji']} {cat_name}：</span>
            <span>{items_str}</span>
        </div>
        """
    html += '</div>'
    return html

def main():
    model, device = load_model()

    with gr.Blocks(title="垃圾分类识别系统", css=CUSTOM_CSS) as demo:

        # 导航栏
        gr.HTML("""
        <div class="navbar">
            <div class="navbar-brand">
                <div class="navbar-logo">🗑️</div>
                <span class="navbar-title">垃圾分类识别</span>
            </div>
        </div>
        """)

        # --- 识别核心区 ---
        with gr.Row(equal_height=True):
            # 左侧：上传
            with gr.Column(scale=1, elem_classes=["product-panel"]):
                gr.HTML('<div class="panel-title-bar"><span class="panel-title-text">📤 上传垃圾图片</span></div>')
                image_input = gr.Image(type="numpy", sources=["upload", "webcam"], show_label=False, container=False)

            # 右侧：结果
            with gr.Column(scale=1, elem_classes=["product-panel"]):
                gr.HTML('<div class="panel-title-bar"><span class="panel-title-text">🎯 识别结果</span></div>')
                result_output = gr.Label(num_top_classes=5, show_label=False, container=False)

        # 间隔空间
        gr.HTML('<div style="margin-top: 32px;"></div>')

        # --- 下方直接平铺展示具体分类 ---
        gr.HTML(build_guide_html())

        # 事件绑定
        image_input.change(
            fn=lambda img: predict(img, model, device),
            inputs=image_input,
            outputs=result_output
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == '__main__':
    main()