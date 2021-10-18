clc
clear all
close all

rng(1234)

n=65;
A=zeros(n, n);

topLayer = 1:5;
middleLayer = 6:20;
bottomLayer = 21:65;

pairs_filename='pairs_golden_standard';
pairs_wid=fopen(pairs_filename, 'w');

self_filename='self_golden_standard';
self_wid=fopen(self_filename, 'w');

for i=topLayer
    A(i, (i*3+3):(i*3+5))=1;
end

% for i=middleLayer
%     A(i, (i*3+3):(i*3+5))=1;
% end

for i=1:5
    temp1 = randi(5);
    temp2 = randi(15)+5;
    A(temp1, temp2) = 1;
end

for i=1:15
    temp1 = randi(15)+5;
    temp2 = randi(15)+5;
    if (temp1 == temp2)
        continue;
    end
    A(temp1, temp2) = 1;
end

% for i=1:45
%     temp1 = randi(15)+5;
%     temp2 = randi(45)+20;
%     A(temp1, temp2) = 1;
% end

for i = size(A,1)
    A(i,i) = 0;
end


array=sum(A);
for i=1:length(array)
    if array(i)==0
        fprintf(self_wid, '%d\t%d\t%d\t%f\n', i, i, 1, 0.95+0.05*rand());
    end
end

for i=1:size(A,1)
    for j=1:size(A,2)
        if i~=j && A(i, j)~=0
            fprintf(pairs_wid, '%d\t%d\t%d\t%f\n', i, j, randi(3), -1+2*rand()); 
        end
    end
end


imagesc(A)
grid on
colorbar
save('groundtruth.mat', 'A')

fclose(pairs_wid);
fclose(self_wid);
