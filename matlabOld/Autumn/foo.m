t = 10.^(-5:1:5)';
y1=t;
y2=t.*log(t);
plot(t,y1);
hold on
plot(t,y2,'Linewidth',2);
