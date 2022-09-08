# Fine-Grained Image Analysis with Deep Learning: A Survey

### Abstract

- 细粒度图像分析 (FGIA) 是计算机视觉和模式识别中一个长期存在的基本问题，并支撑着一系列不同的现实世界应用。
-  FGIA 的任务目标是分析从属类别中的视觉对象，例如鸟类种类或汽车模型。细粒度图像分析固有的小类间和大类内变化使其成为一个具有挑战性的问题。近年来，利用深度学习的进步，我们见证了深度学习驱动的 FGIA 取得了显着进展。
- 在本文中，我们对这些进展进行了系统的调查，我们试图通过合并两个基本的细粒度研究领域——细粒度图像识别和细粒度图像检索来重新定义和拓宽 FGIA 领域。
- 此外，我们还回顾了 FGIA 的其他关键问题，例如公开可用的基准数据集和相关的特定领域应用程序。
- 最后，我们强调了几个研究方向和需要社区进一步探索的开放问题

![image-20220906143010192](TyporaImg/image-20220906143010192.png)

### Recognition

- 我们将不同类型的细粒度识别方法组织成三种范式，即 
  - 1）通过定位分类子网络进行识别，
  - 2）通过端到端特征编码进行识别，以及 
  - 3）利用外部信息进行识别.
  - 细粒度识别是FGIA中研究最多的领域，因为识别是大多数视觉系统的基本能力，值得长期持续研究。
- 细粒度图像分析 (FGIA) 侧重于处理属于**同一元类别的多个从属类别的对象**（例如，不同种类的鸟类或不同型号的汽车），通常涉及两个中心任务：**细粒度图像识别和细粒度的图像检索**。如图 3 所示，细粒度分析位于基本类别分析（即通用图像分析）和实例级分析（例如个体识别）之间的连续统一体。
- 具体来说，FGIA 与通用图像分析的区别在于，在通用图像分析中，目标对象属于粗粒度的元类别（即基本级别的类别），因此在视觉上完全不同（例如，确定图像是否包含鸟、水果或狗）。
  然而，在 FGIA 中，由于对象通常来自同一元类别的子类别，因此问题的细粒度性质导致它们在视觉上相似。作为细粒度识别的一个例子，在图 1 中，任务是对不同品种的狗进行分类。为了准确识别图像，有必要捕捉细微的视觉差异（例如，辨别特征，如耳朵、鼻子或尾巴）。其他 FGIA 任务（例如，检索）也需要表征这些特征。此外，如前所述，该问题的细粒度性质具有挑战性，因为高度相似的子类别引起的小类间变化，以及姿势、尺度和旋转的大类内变化（见图 4） .它与通用图像分析相反（即小的类内变化和大的类间变化），是什么使 FGIA 成为一个独特且具有挑战性的问题。

![image-20220906144208388](TyporaImg/image-20220906144208388.png)

- 小的类间变化和大的类内变化

### Recognition by Localization-Classification Subnetworks

![image-20220906144805487](TyporaImg/image-20220906144805487.png)

- 使用检测或者分割方法
  - 直接使用检测或分割技术 [116]、[117]、[118] 来定位与细粒度对象部分相对应的关键图像区域，例如，鸟头、鸟尾、车灯、狗耳朵、狗躯干、等由于定位信息，即部分级别的边界框或分割掩码，该模型可以获得更具辨别力的中间级别（部分级别）表示
- 使用深度滤波器
  - 研究人员逐渐发现中间 CNN 输出（例如，局部深度描述符）可以与常见对象的语义部分相关联 [122]。因此，细粒度社区试图将这些滤波器输出用作部件检测器 [76]、[77]、[78]、[79]、[80]、[81]、[82]，从而依赖它们来检测进行定位-分类细粒度识别。
- 注意力机制

### Recognition by End-to-End Feature Encoding

- 细粒度识别的第二种学习范式是端到端的特征编码。与其他视觉任务一样，特征学习在细粒度识别中也发挥着重要作用。由于子类别之间的差异通常非常微妙和局部，因此仅使用全连接层捕获全局语义信息会限制细粒度模型的表示能力，从而限制最终识别性能的进一步提高。因此，已经开发了一些方法，旨在通过以下方式学习统一但有区别的图像表示，以对细粒度类别之间的细微差异进行建模
- 通过执行高阶特征交互
  - SIFT
- 通过设计新颖的损失函数
- 通过其他方式

### Recognition with External Information
