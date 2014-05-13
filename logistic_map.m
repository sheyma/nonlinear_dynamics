Npre = 5000; Nplot = 300;
x = zeros(Nplot,1);
for r = 2.5:0.0001:3.7,
  x(1) = 0.5;
  for n = 1:Npre,
    x(1) = r*x(1)*(1 - x(1));
  end,
  for n = 1:Nplot-1,
    x(n+1) = r*x(n)*(1 - x(n));
  end,
  plot(r*ones(Nplot,1), x, '.', 'markersize', 2);
  hold on;
end,
set(gca,'FontSize',25)
title('Bifurcation diagram of the logistic map');
xlabel('r');  ylabel('x_n');
r1 = 2.998;
r2 = 3.449;
r3 = 3.544;
r4 = 3.564;
r5=3.569;

line([r1 r1],[0.2 1],'Color','r','LineWidth',0.2)
line([r2 r2],[0.2 1],'Color','r','LineWidth',0.2)
line([r3 r3],[0.2 1],'Color','r','LineWidth',0.2)
line([r4 r4],[0.2 1],'Color','r','LineWidth',0.2)
line([r5 r5],[0.2 1],'Color','r','LineWidth',0.2)
set(gca, 'xlim');
axis([2.5, 3.7, 0.2 , 1])
hold off; 

r = 0:0.001:4;
a = length(r);

lyap=zeros(1,N);
j=0;
for(r=0:0.001:4)
    xn1=rand(1);
    lyp=0;
    j=j+1;
    for(i=1:10000)
        xn=xn1;
        %logistic map
        xn1=r*xn*(1-xn);
       %wait for transient
       if(i>300)
           % calculate teh sum of logaritm
           lyp=lyp+log(abs(r-2*r*xn1));
       end
    end
    %calculate lyapun
    lyp=lyp/10000;
    lyap(j)=lyp;
end
r = 0:0.001:4;
plot(r,lyap);
