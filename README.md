# PSGAN-NCNN

1. forward1出来的gamma和beta没有底数，应该是乘200没把底值拉起来
2. 添加了opencv + dlib + ncnn的vs命令行实现代码
3. 需要再检查一遍预处理代码，实在不行就放弃了，放个半成品代码出去吧
4. 效果能改回来的话，把dlib去了，用两个ncnn模型替代，opencv就可以换成opencv-mobile，然后用qt做个界面，计划通