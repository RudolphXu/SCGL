function [expression] = standalization(expression)
%STANDALIZATION Summary of this function goes here
%   Detailed explanation goes here
    n=size(expression, 1);
    T=size(expression, 2);

    aver=repmat(mean(expression'), [T,1]);
    dev=repmat(std(expression'), [T, 1]);

    expression=(expression-aver')./dev';

    expression = expression';
end

