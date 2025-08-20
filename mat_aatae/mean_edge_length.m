function m = mean_edge_length(g,H)
    e=[]; for ii=1:size(H,1), c=nchoosek(H(ii,:),2); e=[e;c]; end %#ok<AGROW>
    e = unique(sort(e,2),'rows');
    L = sqrt(sum((g(e(:,1),:)-g(e(:,2),:)).^2,2)); m = mean(L);
end