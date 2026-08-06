# S4 线性验证政策变更

旧方法：
用比有限差分 epsilon 大 50～1000 倍的区域，直接作为平衡点 Jacobian 的局部通过门槛。

问题：
该测试衡量的是较宽运行区域的一阶近似能力，不能单独判断平衡点 Jacobian 是否正确。

新方法：
1. 中心有限差分重复性与半 epsilon 收敛；
2. 10×epsilon 真正局部验证作为 Jacobian 验收；
3. 原宽范围验证完整保留为 operating-region limitation；
4. 最终以真实非线性三场景安全性、位置公平性和摆动改善作为控制器验收。

说明：`model_hinge_frictionloss=0.005` 已记录在 `validation_scale_audit.json`。摩擦可能导致非光滑性，但没有对照实验时不宣称其为唯一原因。
