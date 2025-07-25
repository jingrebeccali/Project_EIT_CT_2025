function allFiles = collectAllSlices(baseFolder)
% Parcourt chaque sujet et liste tous les fichiers avec leur z
  d = dir(baseFolder);
  subs = d([d.isdir] & ~ismember({d.name},{'.','..'}));
  allFiles = struct('subject',{},'file',{},'z',[]);
  cnt = 0;
  for i=1:numel(subs)
    subj = subs(i).name;
    files = dir(fullfile(baseFolder,subj,'*_mesh_slice_*.mat'));
    for k=1:numel(files)
      S = load(fullfile(baseFolder,subj,files(k).name),'z_min');
      cnt=cnt+1;
      allFiles(cnt).subject = subj;
      allFiles(cnt).file    = files(k).name;
      allFiles(cnt).z       = S.z_min;
    end
  end
end


function [zoneIdx, edges] = assignZonesUniform(zValues,K)
  edges   = linspace(min(zValues),max(zValues),K+1);
  zoneIdx = discretize(zValues, edges);
end

function zoneIdx = assignZonesKmeans(zValues,K)
  zoneIdx = kmeans(zValues(:), K, 'Replicates',5);
end















% function results = computeBestSlicesAllSubjects(baseFolder, refSubject, refSliceNum, N)
% % Pour chaque sujet (sous-dossier) ≠ refSubject, sélectionne les N slices
% %
% % results(i) a
% %   .subject       nom du sujet (ex. 's0059')
% %   .bestIdxSlices vecteur des slice numbers choisis
% %   .bestFileNames cell array des fichiers .mat correspondants
% 
%     %Précharge le mesh de référence
%     refDir  = fullfile(baseFolder, refSubject);
%     refFile = sprintf('%s_mesh_slice_%d.mat', refSubject, refSliceNum);
%     S       = load(fullfile(refDir, refFile));
%     g_ref   = S.g(:,1:min(2,end));  
% 
%     D    = dir(baseFolder);
%     isSD = [D.isdir] & ~ismember({D.name},{'.','..'});
%     subs = D(isSD);
% 
%     results = struct('subject',{},'bestIdxSlices',{},'bestFileNames',{});
%     idxRes = 1;
%     for i = 1:numel(subs)
%         subj = subs(i).name;
%         if strcmp(subj, refSubject)
%             continue
%         end
%         subjDir = fullfile(baseFolder, subj);
% 
%         % Calcule les meilleures slices pour ce sujet
%         [bestIdx, bestFiles] = getBestSlicesForDir(subjDir, g_ref, N);
% 
%         % Stocke
%         results(idxRes).subject       = subj;
%         results(idxRes).bestIdxSlices = bestIdx;
%         results(idxRes).bestFileNames = bestFiles;
%         idxRes = idxRes + 1;
%     end
% end
% 
% 
% 
% %%%%%%%%%%
% function [bestIdxSlices, bestFileNames] = getBestSlicesForDir(Dir, g_ref, N)
% %GETBESTSLICESFORDIR   Renvoie les N indices de slice et leurs fichiers .mat les plus proches
% %
% %   Dir        : chemin vers un sous‑dossier sXXXX
% %   g_ref      : points de référence (NA×2)
% %   N          : nombre de slices à retenir
% 
% 
%     patt = '^s\d{4}_mesh_slice_(\d+)\.mat$';
%     F0   = dir(fullfile(Dir,'*.mat'));
% 
%     sliceNums    = [];
%     errs         = [];
%     fileNames    = {};
% 
%     for k = 1:numel(F0)
%         name = F0(k).name;
%         tok  = regexp(name, patt, 'tokens', 'once');
%         if isempty(tok)
%             continue  % pas un slice valide
%         end
% 
% 
%         z = str2double(tok{1});
%         fullPath = fullfile(Dir, name);
%         try
%             S = load(fullPath);
%         catch ME
%             warning('Skip file %s (load error: %s)', name, ME.message);
%             continue
%         end
% 
%         % Récupération et projection 2D du nuage
%         if ~isfield(S,'g')
%             warning('Skip file %s (pas de champ g)', name);
%             continue
%         end
%         g = S.g;
%         if size(g,2) > 2
%             g = g(:,1:2);
%         end
% 
%         %Calcul de la distance
%         try
%             d = hausdorff_distance(g_ref, g);
%         catch ME
%             warning('Erreur de distance sur %s : %s', name, ME.message);
%             continue
%         end
% 
%         % Stockage
%         sliceNums(end+1,1) = z;    
%         errs(end+1,1)      = d;    
%         fileNames{end+1,1} = name;
%     end
% 
%     %Si aucun fichier valide
%     if isempty(errs)
%         bestIdxSlices  = [];
%         bestFileNames  = {};
%         return
%     end
% 
%     % Tri  et sélection des N premiers slices
%     [~, order]     = sort(errs);
%     cnt            = min(N, numel(order));
%     sel            = order(1:cnt);
% 
%     bestIdxSlices  = sliceNums(sel);
%     bestFileNames  = fileNames(sel);
% end
