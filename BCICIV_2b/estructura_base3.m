clear all
for in=1:3
    nombre=sprintf('B010%dT.gdf',in);
    [s, h] = sload(nombre, 0, 'OVERFLOWDETECTION:OFF');
    [a, ~] = find(h.out.EVENT.TYP==1023);
    h.out.EVENT.TYP(a+1)=[];
    h.out.EVENT.POS(a+1)=[];
    clear a, clear pos
    [a, ~] = find(h.out.EVENT.TYP==769);
    pos=h.out.EVENT.POS(a)+376;
    if exist('C1')
        lag=size(C1,3);
    else
        lag=0;
    end
    for i=1:length(pos)
        C1(:,:,i+lag)=s(pos(i):pos(i)+499,1:3)';
    end
    
    clear pos
    clear a
    [a, ~] = find(h.out.EVENT.TYP==770);
    pos=h.out.EVENT.POS(a)+376;
    if exist('C2')
        lag2=size(C2,3);
    else
        lag2=0;
    end
    for i=1:length(pos)
        C2(:,:,i+lag2)=s(pos(i):pos(i)+499,1:3)';
    end   
end
% save('S9.mat','C1', 'C2')

    