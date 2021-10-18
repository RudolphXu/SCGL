clc
clear all
close all

rng(1234)


n=130;
A=zeros(n, n);

topLayer = 1:10;
middleLayer = 11:40;
bottomLayer = 41:130;

pairs_filename='pairs_golden_standard';
pairs_wid=fopen(pairs_filename, 'w');

self_filename='self_golden_standard';
self_wid=fopen(self_filename, 'w');

for i=topLayer
    A(i, (i*3+8):(i*3+10))=1;
end

for i=middleLayer
    A(i, (i*3+8):(i*3+10))=1;
end


for i=1:10
    temp1 = randi(10);
    temp2 = randi(30)+10;
    A(temp1, temp2) = 1;
end

for i=1:30
    temp1 = randi(30)+10;
    temp2 = randi(30)+10;
    if (temp1 == temp2)
        continue;
    end
    A(temp1, temp2) = 1;
end

for i=1:90
    temp1 = randi(30)+10;
    temp2 = randi(90)+40;
    A(temp1, temp2) = 1;
end

for i = size(A,1)
    A(i,i) = 0;
end


array=sum(A);
for i=1:length(array)
    if array(i)==0
        % Half of the master nodes are not activated
        fprintf(self_wid, '%d\t%d\t%d\t%f\n', i, i, 1, 0.95+0.05*rand());
    end
end


for i=1:size(A,1)
    for j=1:size(A,2)
        if i~=j && A(i, j)~=0
            fprintf(pairs_wid, '%d\t%d\t%d\t%f\n', i, j, randi(5), -1+2*rand()); 
        end
    end
end


imagesc(A)

save('groundtruth.mat', 'A')

fclose(pairs_wid);
fclose(self_wid);
