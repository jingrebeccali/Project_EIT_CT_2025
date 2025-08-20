function warpedResults = warpAndComputeMeasurements(...
    baseFolder, results, refSubject, refSliceNum, ...
    Ncontour, n_el, L, dataQ)
% Fonction parallèle, mais *sans* ouvrir/fermer le pool
% IN
%   dataQ : parallel.pool.DataQueue pour logs (ou [] si pas de logs)

  if nargin<8, dataQ = []; end

  % 1) Charge et prépare la slice de référence
  refDir   = fullfile(baseFolder, refSubject);
  refFile  = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
  S_ref    = load(fullfile(refDir,refFile), 'g','H','sigma','edges');
  g_ref    = S_ref.g(:,1:2);
  H_ref    = S_ref.H + 1;
  sig_ref  = S_ref.sigma;
  edges_ref= S_ref.edges;

  % électrodes + pattern courant
  %%E_ref = generateElectrodesSurfaces(g_ref, edges_ref, n_el, L);
  M     = eye(n_el) - circshift(eye(n_el), [0,n_el/2]);
  injV  = M(:);

  % contour de référence pour TPS
  [~, bndA] = extract_ordered_boundary(g_ref, edges_ref);
  dst       = resample_contour_by_arclength(bndA, Ncontour);

  % 2) Aplatit la liste des tâches
  total = sum(arrayfun(@(s) numel(s.bestIdxSlices), results));
  tasks(total) = struct('subject','','slice',[],'file','');
  idx = 0;
  for i = 1:numel(results)
    subj = results(i).subject;
    sls  = results(i).bestIdxSlices;
    fls  = results(i).bestFileNames;
    for j = 1:numel(sls)
      idx = idx + 1;
      tasks(idx).subject = subj;
      tasks(idx).slice   = sls(j);
      tasks(idx).file    = fls{j};
    end
  end

  % 3) Pré‑allocation pour recueillir les résultats
  subjCell  = cell(total,1);
  sliceList = zeros(total,1);
  ImeasCell = cell(total,1);
  sigCell   = cell(total,1);
  % 4) Boucle parfor pure
  parfor t = 1:total
    tk = tasks(t);

    % 4.1) Charge la slice cible
    S = load(fullfile(baseFolder, tk.subject, tk.file), ...
             'g','H','sigma','edges','z','z_min','present_organs');
    gB    = S.g(:,1:2);
    HB    = S.H + 1;
    sigB  = S.sigma;
    edB   = S.edges+1;
    znum  = S.z- S.z_min;
    
    % 4.2) TPS align + warp
    [~, bndB] = extract_ordered_boundary(gB, edB);
    src       = resample_contour_by_arclength(bndB, Ncontour);
    [src_a,~,~] = align_contours(src, dst);
    tps       = ThinPlateSpline2D().fit(src_a, dst);
    nodesW    = tps.transform(gB);
    E = generateElectrodesSurfaces(nodesW,edB,n_el,L);
    % 4.3) Solveur CEM
    fmeshW    = ForwardMesh1st(nodesW, HB, E);
    solverW   = EITFEM(fmeshW);
    solverW.mode = 'current';
    solverW.Iel  = injV;
    sig_real = Sig_update(S.present_organs,sigB);

    Iw        = solverW.SolveForwardVec(sig_real);

    % 4.4) Log immédiat via DataQueue
    if ~isempty(dataQ)
      send(dataQ, struct('subject',tk.subject,'slice',znum));
    end

    % 4.5) Stockage
    subjCell{t}   = tk.subject;
    sliceList(t)  = znum;
    ImeasCell{t}  = Iw;
    sigCell{t}    = sig_real;
  end

  %5) Assemble le struct de sortie
  C = [subjCell, num2cell(sliceList), ImeasCell,sigCell];
  warpedResults = cell2struct(C, {'subject','slice','Imeas_wrapped','sigCell'}, 2);

  % 6) Récapitulatif par sujet (séquentiel)
  subsDone = unique({warpedResults.subject});
  for i = 1:numel(subsDone)
    cnt = sum(strcmp({warpedResults.subject}, subsDone{i}));
    
  end
end



% function warpedResults = warpAndComputeMeasurements(baseFolder, results, refSubject, refSliceNum, Ncontour,n_el,L)
% %WARPANDCOMPUTEMEASUREMENTS  TPS‑warp + forward solveur pour chaque slice sélectionnée
% %
% % warpedResults = warpAndComputeMeasurements(baseFolder, results, ...
% %                   refSubject, refSliceNum, Ncontour)
% %
% % IN
% %   baseFolder   : dossier parent contenant tous les sous‑dossiers sXXXX
% %   results      : struct array retourné par
% %                  computeBestSlicesAllSubjects, avec champs
% %                     .subject, .bestIdxSlices, .bestFileNames
% %   refSubject   : nom du sujet de référence, ex. 's0011'
% %   refSliceNum  : numéro de slice de référence, ex. 190
% %   Ncontour     : nombre de points pour le rééchantillonnage des contours (p.ex. 400)
% %
% % OUT
% %   warpedResults : struct array avec champs :
% %      .subject        nom du sujet
% %      .slice          numéro de slice
% %      .Imeas_wrapped  vecteur colonne des mesures retournées par SolveForwardVec
% 
%   %%% prépare le mesh de référence  (  !!!!! Ce calcul est fait pour la
%   %%%  3-ième fois !!!!!!!!   TO DO ) 
%   refDir   = fullfile(baseFolder, refSubject);
%   refFile  = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
%   S_ref    = load(fullfile(refDir, refFile));
%   g_ref    = S_ref.g;  
%   H_ref    = S_ref.H + 1;
%   sig_ref  = S_ref.sigma;
%   edges_ref= S_ref.edges;
%   if size(g_ref,2) > 2
%     g_ref = g_ref(:,1:2);
%   end
%   % Electrodes de référence (suppose n_el constant)
%   % E_ref = cell(n_el,1);
%   % for L = 1:n_el
%   %   fld = sprintf('E%d', L);
%   %   E_ref{L} = S_ref.(fld) + 1;
%   % end
%   E_ref = generateElectrodesSurfaces(g_ref, edges_ref, n_el, L);
% 
% 
%   [~, bnd_pts_A] = extract_ordered_boundary(g_ref, edges_ref);
%   dst = resample_contour_by_arclength(bnd_pts_A, Ncontour);
% 
%   %%%%  boucle sur tous les sujets/slices
%   total = sum(arrayfun(@(s) numel(s.bestIdxSlices), results));
%   warpedResults(total) = struct('subject','','slice',[],'Imeas_wrapped',[]);
% 
%   idxOut = 0;
%   for iSub = 1:numel(results)
%     subjName   = results(iSub).subject;
%     slicesList = results(iSub).bestIdxSlices;
%     fileList   = results(iSub).bestFileNames;
%     subjDir    = fullfile(baseFolder, subjName);
% 
%     for j = 1:numel(slicesList)
%       z   = slicesList(j);
%       fn  = fileList{j};
%       idxOut = idxOut + 1;
%       S     = load(fullfile(subjDir, fn));
%       gB    = S.g; HB = S.H + 1; edgesB = S.edges; sigmaB = S.sigma;
%       if size(gB,2) > 2
%         gB = gB(:,1:2);
%       end
% 
%       [~, bnd_pts_B] = extract_ordered_boundary(gB, edgesB);
%       src = resample_contour_by_arclength(bnd_pts_B, Ncontour);
% 
%       [src_aligned, ~, ~] = align_contours(src, dst);
% 
%       tps = ThinPlateSpline2D();
%       tps = tps.fit(src_aligned, dst);
% 
%       nodesA_warp = tps.transform(gB);
% 
%       % Construction du solveur et calcul des mesures
%       fmesh_wrapped   = ForwardMesh1st(nodesA_warp, HB, E_ref);
%       solver_wrapped  = EITFEM(fmesh_wrapped);
%       solver_wrapped.mode = 'current';
%       M = eye(n_el) - circshift(eye(n_el), [0, n_el/2]);
% 
%       solver_wrapped.Iel  = M(:);
%       Imeas_wrapped   = solver_wrapped.SolveForwardVec(sigmaB);
%     %   [Eset, Imeas_wrapped] = computeMeasurementsOverRotations( ...
%     % nodesA_warp, HB, edges_ref, sigmaB, n_el, L, M(:));
% 
%       % 2.7) Stockage du résultat
%       warpedResults(idxOut).subject       = subjName;
%       warpedResults(idxOut).slice         = z- S.z_min;
%       warpedResults(idxOut).Imeas_wrapped = Imeas_wrapped;
%       fprintf('la mes de la slice %d du sujet %s is done \n',z,subjName);
% 
%     end
%     fprintf('totues les mes du sujet %s is done \n',subjName);
% 
%   end
% end
