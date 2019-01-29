[toc]

# 理论介绍

## 什么是分类

- 分类属于机器学习中监督学习的一种。模型的学习在被告知每个训练样本属于哪个类的“指导”下进行，新数据使用训练集中得到的规则进行分类。
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznh9w9bkkj21bb0lpdjp.jpg)

## 分类的步骤

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhce62gej219w0g7q6r.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhce2eapj21930my786.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhce1axkj216i0hqdjs.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhce2pr0j21670myq6d.jpg)

## 什么是决策树

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhdty6snj217e0najuu.jpg)

## 决策树归纳

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhejo49mj21aq0kkq7w.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhejh77dj21aj0dn0uu.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhejssrxj21980msgs1.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhejsansj21aq0medjz.jpg)

## 信息增益

### 相关理论基础

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhf5v2dqj219z0n579q.jpg)

### 计算公式
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhfpkhl2j21am0n4jx6.jpg)

## ID3

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhh84f09j21a80nxtca.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhhp68bij217c0nbdjx.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhhp6rqlj217m0mxtcc.jpg)
![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhhp1ldrj212t0jiwg2.jpg)
**注：生成的决策树有误，fair对应的应该是yes,excellent对应的应该是no**

## C4.5

![](https://ws1.sinaimg.cn/large/874b0eb1gy1fznhjljoypj21bp0ledla.jpg)

# python实现

- [GitHub地址](https://github.com/Professorchen/Machine-Learning)

# 参考资料

- 理论部分参考：福州大学数学与计算机科学学院苏雅茹老师数据挖掘课上使用的课件（一并上传到Github了，课件内还包括其他一些常用分类算法，例如贝叶斯算法）
- 代码部分参考：[Python实现C4.5(信息增益率)](https://www.cnblogs.com/wsine/p/5180315.html)
- **如有侵权，请联系我删除**