import os
import sys
import glob

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from tgsr_inference import TGSRPredictor

def main():
    # 设置模型路径
    model_dir = '/root/autodl-tmp/TGSR/experiments/train_TGSRx4plus_400k_B12G4/models'
    
    try:
        # 创建预测器
        print("正在加载模型...")
        predictor = TGSRPredictor(
            sr_model_path=os.path.join(model_dir, 'net_g_100000.pth'), 
            text_guidance_path=os.path.join(model_dir, 'net_t_100000.pth'),
            text_encoder_path='/root/autodl-tmp/clip-vit-base-patch32'
        )
        
        # 查找测试图像
        print("查找测试图像...")
        test_dirs = [
            '/root/autodl-tmp/TGSR/tgsr',
        ]
        
        img_path = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_images = glob.glob(os.path.join(test_dir, f'*{ext}'))
                    if potential_images:
                        img_path = potential_images[0]
                        break
                if img_path:
                    break
        
        # 如果仍找不到图像，查找任何图像
        if not img_path:
            print("在预定义目录中找不到图像，尝试查找任何可用图像...")
            potential_images = []
            for ext in ['.png', '.jpg', '.jpeg']:
                found = glob.glob(f'/root/autodl-tmp/**/*{ext}', recursive=True)
                if found:
                    potential_images.extend(found[:5])  # 限制数量
            
            if potential_images:
                img_path = potential_images[0]
        
        if not img_path:
            print("错误：找不到任何测试图像")
            return
            
        # 创建输出目录
        output_dir = '/root/autodl-tmp/TGSR/results'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"处理图像: {img_path}")
        
        # 执行推理
        result = predictor.predict(
            img_path=img_path,
            text="fuck you",
            output_dir=output_dir,
            save_attention=True
        )
        
        print(f"处理完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()