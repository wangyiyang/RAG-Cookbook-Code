"""
RAGAS评估演示和使用示例
展示如何使用RAGAS快速评估RAG系统质量
"""

from ragas_evaluator import RAGASEvaluator, quick_evaluate, batch_evaluate
import json
import time


def demo_basic_evaluation():
    """基础评估演示"""
    print("=== 基础RAGAS评估演示 ===\n")
    
    # 准备测试数据 - 这是你需要准备的格式
    test_data = [
        {
            'question': 'Python中如何创建虚拟环境？',
            'answer': '在Python中创建虚拟环境可以使用venv模块。命令是：python -m venv myenv，然后用source myenv/bin/activate激活。',
            'contexts': [
                'Python虚拟环境可以使用venv模块创建，命令为python -m venv 环境名称。',
                '激活虚拟环境的命令在Linux/Mac下是source venv/bin/activate，Windows下是venv\\Scripts\\activate。'
            ],
            'ground_truth': 'Python创建虚拟环境使用python -m venv命令，激活用source venv/bin/activate。'
        },
        {
            'question': '什么是Docker容器？',
            'answer': 'Docker容器是一种轻量级的虚拟化技术，它将应用程序及其依赖打包在一个可移植的容器中。容器提供了隔离的运行环境，确保应用在不同环境中的一致性。',
            'contexts': [
                'Docker是一个开源的容器化平台，用于开发、发布和运行应用程序。',
                '容器是一种轻量级的虚拟化技术，相比传统虚拟机消耗更少资源。',
                '容器提供了应用程序运行所需的所有依赖，包括系统库、工具和运行时环境。'
            ],
            'ground_truth': 'Docker容器是轻量级虚拟化技术，用于打包和运行应用程序，提供隔离的运行环境。'
        }
    ]
    
    # 执行评估
    print("正在执行RAGAS评估...")
    result = batch_evaluate(test_data)
    
    # 输出结果
    print(f"✅ 评估完成！")
    print(f"📊 综合评分: {result['overall_score']:.3f}")
    print(f"🏆 质量等级: {result['quality_grade']}")
    
    print("\n📈 详细指标:")
    for metric, score in result['scores'].items():
        print(f"  - {metric}: {score:.3f}")
    
    print("\n💡 优化建议:")
    for i, rec in enumerate(result['recommendations'], 1):
        priority_icon = {"high": "🔥", "medium": "⚠️", "low": "💡", "info": "ℹ️"}.get(rec['priority'], "")
        print(f"  {i}. {priority_icon} {rec['issue']}")
        print(f"     解决方案: {rec['solution']}")


def demo_quick_evaluation():
    """快速单次评估演示"""
    print("\n\n=== 快速单次评估演示 ===\n")
    
    # 单次快速评估
    print("正在进行快速评估...")
    result = quick_evaluate(
        question="什么是机器学习？",
        answer="机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测。",
        contexts=[
            "机器学习是人工智能(AI)的一个分支，专注于构建能够从数据中学习的系统。",
            "机器学习算法通过分析大量数据来识别模式，并使用这些模式进行预测。",
            "与传统编程不同，机器学习让计算机能够从经验中自动改进性能。"
        ],
        ground_truth="机器学习是人工智能的子领域，使计算机系统能够从数据中自动学习和改进性能。"
    )
    
    print(f"⚡ 快速评估结果: {result['overall_score']:.3f}")
    print(f"🎯 质量等级: {result['quality_grade']}")


def demo_problematic_cases():
    """问题案例演示 - 展示低分情况"""
    print("\n\n=== 问题案例演示 ===\n")
    
    # 故意设计一些有问题的案例
    problematic_data = [
        {
            'question': 'Python如何安装第三方库？',
            'answer': 'Python安装第三方库最好的方法是使用JavaScript的npm包管理器。你只需要运行npm install 包名即可。另外，Python还支持Java的Maven管理依赖。',  # 完全错误的答案
            'contexts': [
                'Python使用pip工具来安装第三方库，命令格式为pip install 包名。',
                'pip是Python的包管理工具，随Python一起安装。',
                '也可以使用conda来管理Python包，特别适合数据科学项目。'
            ],
            'ground_truth': 'Python使用pip工具安装第三方库，命令是pip install 包名。'
        },
        {
            'question': '什么是深度学习？',
            'answer': '今天天气不错。',  # 完全不相关的答案
            'contexts': [
                '深度学习是机器学习的一个分支，使用人工神经网络进行学习。',
                '深度学习模型具有多个隐藏层，能够学习数据的层次化特征表示。'
            ],
            'ground_truth': '深度学习是机器学习的子集，使用深层神经网络模拟人脑学习过程。'
        }
    ]
    
    # 评估问题案例
    print("正在评估问题案例...")
    result = batch_evaluate(problematic_data)
    
    print(f"❌ 问题案例评分: {result['overall_score']:.3f}")
    print(f"⚠️  质量等级: {result['quality_grade']}")
    
    print("\n🚨 发现的问题:")
    for rec in result['recommendations']:
        if rec['priority'] in ['high', 'medium']:
            print(f"  - {rec['issue']}")
            print(f"    {rec['solution']}")


def demo_version_comparison():
    """版本对比演示"""
    print("\n\n=== 版本对比演示 ===\n")
    
    # 模拟两个版本的RAG系统结果
    version_1_data = [
        {
            'question': 'Python中如何处理异常？',
            'answer': '使用try-except语句。',  # 简短回答
            'contexts': [
                'Python使用try-except语句来处理异常。',
                '可以使用finally子句来执行清理代码。',
                'raise语句可以主动抛出异常。'
            ]
        }
    ]
    
    version_2_data = [
        {
            'question': 'Python中如何处理异常？', 
            'answer': 'Python使用try-except语句处理异常。在try块中编写可能出错的代码，在except块中处理异常。还可以使用finally子句执行清理操作，使用else子句在没有异常时执行代码。',  # 更详细的回答
            'contexts': [
                'Python使用try-except语句来处理异常。',
                '可以使用finally子句来执行清理代码。',
                'raise语句可以主动抛出异常。'
            ]
        }
    ]
    
    # 创建评估器并对比版本
    evaluator = RAGASEvaluator()
    comparison = evaluator.compare_versions(
        version_1_data, 
        version_2_data, 
        names=("简化版本", "详细版本")
    )
    
    print("📊 版本对比结果:")
    print(f"  简化版本综合得分: {comparison['overall_comparison']['简化版本']:.3f}")
    print(f"  详细版本综合得分: {comparison['overall_comparison']['详细版本']:.3f}")
    
    print("\n📈 指标对比:")
    for metric, data in comparison['scores_comparison'].items():
        change_icon = {"improved": "⬆️", "regressed": "⬇️", "stable": "➡️"}.get(data['change'], "")
        print(f"  {metric}: {data['简化版本']:.3f} → {data['详细版本']:.3f} {change_icon}")
    
    if comparison['improvements']:
        print(f"\n✅ 改进项: {', '.join(comparison['improvements'])}")
    if comparison['regressions']:
        print(f"\n❌ 退步项: {', '.join(comparison['regressions'])}")


def demo_custom_evaluation():
    """自定义评估演示"""
    print("\n\n=== 自定义评估演示 ===\n")
    
    # 创建自定义阈值的评估器
    evaluator = RAGASEvaluator()
    
    # 修改评估阈值（针对特定业务场景）
    evaluator.quality_thresholds = {
        'faithfulness': 0.9,        # 对忠实度要求更高
        'answer_relevancy': 0.8,    # 对相关性要求更高
        'context_relevancy': 0.7,   # 对检索质量要求更高
        'context_precision': 0.6,   # 对排序要求更高
    }
    
    test_data = [
        {
            'question': '如何优化RAG系统性能？',
            'answer': '优化RAG系统可以从几个方面入手：1）改进检索质量，使用更好的embedding模型；2）优化chunk分割策略；3）实施重排序机制；4）调整生成prompt。',
            'contexts': [
                'RAG系统优化包括检索优化和生成优化两个方面。',
                '检索优化可以通过改进embedding模型、优化chunk策略、实施重排序来实现。',
                '生成优化主要通过prompt engineering和模型fine-tuning。',
                '系统性能监控也很重要，需要定期评估各项指标。'
            ]
        }
    ]
    
    result = evaluator.evaluate(test_data)
    
    print("🎯 高标准评估结果:")
    print(f"  综合评分: {result['overall_score']:.3f}")
    print(f"  质量等级: {result['quality_grade']}")
    
    print("\n📊 在高标准下的表现:")
    for metric, score in result['scores'].items():
        threshold = evaluator.quality_thresholds[metric]
        status = "✅" if score >= threshold else "❌"
        print(f"  {metric}: {score:.3f} (阈值: {threshold}) {status}")


def demo_error_handling():
    """错误处理演示"""
    print("\n\n=== 错误处理演示 ===\n")
    
    # 演示数据格式错误的处理
    invalid_data = [
        {'question': '测试问题'}  # 缺少answer和contexts字段
    ]
    
    try:
        result = batch_evaluate(invalid_data)
    except ValueError as e:
        print(f"❌ 数据格式错误: {e}")
        print("💡 请确保测试数据包含必要字段: question, answer, contexts")
    
    # 演示API配置问题的处理
    print("\n如果遇到API问题，请检查:")
    print("1. 是否设置了OPENAI_API_KEY环境变量")
    print("2. API密钥是否有效")
    print("3. 网络连接是否正常")
    
    # 演示数据质量问题
    print("\n数据质量检查建议:")
    print("- 确保问题表述清晰")
    print("- 检查上下文是否真实相关") 
    print("- 验证答案质量")
    print("- 如果有ground_truth，确保其准确性")


def save_evaluation_results(result: dict, filename: str = "evaluation_result.json"):
    """保存评估结果到文件"""
    
    # 处理不能JSON序列化的对象
    serializable_result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_score': result['overall_score'],
        'quality_grade': result['quality_grade'],
        'scores': result['scores'],
        'recommendations': result['recommendations'],
        'detailed_analysis': result['detailed_analysis']
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print(f"📁 评估结果已保存到: {filename}")


if __name__ == "__main__":
    print("🚀 RAGAS评估演示开始！\n")
    
    # 运行所有演示
    try:
        demo_basic_evaluation()
        demo_quick_evaluation()
        demo_problematic_cases()
        demo_version_comparison()
        demo_custom_evaluation()
        demo_error_handling()
        
        print("\n\n🎉 所有演示完成！")
        print("\n📚 使用建议:")
        print("1. 开发阶段：使用quick_evaluate进行快速验证")
        print("2. 测试阶段：使用batch_evaluate进行全面评估")
        print("3. 对比阶段：使用compare_versions对比不同版本")
        print("4. 生产阶段：定期使用真实数据进行评估")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        print("💡 请检查RAGAS依赖是否正确安装，API配置是否正确")
        print("   安装命令: pip install ragas datasets openai")