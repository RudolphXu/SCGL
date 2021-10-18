clear all
close all
clc
n = 65;

load pairs_golden_standard;
pairs = pairs_golden_standard;
load self_golden_standard;
self = self_golden_standard;


T=603;
expression = zeros(n, T);


rng(200) 
expression(1:n,1)=normrnd(0,1,n,1);
expression(1:n,2)=normrnd(0,1,n,1);
expression(1:n,3)=normrnd(0,1,n,1);


a = exp((1/6)*pi*1.0i);
b = exp(-(1/6)*pi*1.0i);
c = a+b;
d = a*b;

for i=4:T
    for j=1:n
        expression(j,i) = normrnd(0, 1);

        array = find(pairs(:,2)==j);

        if size(array, 1)==0
            index=find(self(:,2)==j);
            expression(j,i) = expression(j,i) +  cos((c*self(index, 4)*expression(j,i-1) * d * self(index,4) * self(index,4) * expression(j,i-2))^2);            

        else
            for k=1:size(array,1)
                expression(j,i) =  expression(j,i) + log((pairs(array(k),4)*expression(pairs(array(k),1), i-pairs(array(k), 3)))^2);            
            end
        end
    end
end

%no noise
expression0 = expression(:,4:end);

[expression0] = standalization(expression0);


figure(1)
plot(expression0)

expression = expression0;
save filter_norm_expression0.mat expression;

