function plotMeshAndElectrodes(g_ref,H_ref,sig_ref,E)

    
    h = trimesh(H_ref(:,1:3), g_ref(:,1), g_ref(:,2), sig_ref);
    set(h, "facecolor", "interp");
    set(h, "facealpha", 0.6);
    set(h, "EdgeColor", "none");
    axis equal;
    grid on;
    % caxis([0,2])
    
    colorbar;
    hold on;
    
    % E is a cell array: E{i} is [Kx2] pairs of boundary node indices
    NE = numel(E);
    cmap = lines(NE);                 % different color per electrode
    
    for i = 1:NE
        % Rebuild the ordered node path of the electrode:
        % edges = [n1 n2; n2 n3; ...] -> nodes = [n1; n2; ...; n(end)]
        ei    = E{i};
        nodes = [ei(:,1); ei(end,2)];
    
        % Draw the electrode as a thick line on the boundary
        plot(g_ref(nodes,1), g_ref(nodes,2), '-', 'LineWidth', 3, 'Color', cmap(i,:));
    
        % Optional: mark the ends
        plot(g_ref(nodes([1 end]),1), g_ref(nodes([1 end]),2), 'o', ...
            'MarkerSize', 4, 'MarkerFaceColor', cmap(i,:), 'Color', cmap(i,:));
    
        % Optional: label roughly at the centroid of the electrode polyline
        cx = mean(g_ref(nodes,1));
        cy = mean(g_ref(nodes,2));
        text(cx, cy, sprintf('%d', i), 'Color', cmap(i,:), ...
            'FontWeight','bold', 'HorizontalAlignment','center', ...
            'VerticalAlignment','middle');
    end
    hold off;
    
    
    
    
    view(2);
end