function sigma_real = Sig_update(orgList,sig)


allNames = { ...
    'soft_tissue','adrenal_gland', 'autochthon', 'clavicula', 'costal_cartilages', ...
    'ribs', 'scapula', 'sternum', 'skull', 'humerus', ...
    'femur', 'vertebrae', 'sacrum', 'gluteus_maximus', ...
    'gluteus_minimus', 'gluteus_medius', 'iliopsoas', 'hip', ...
    'iliac_arterie', 'iliac_vein', 'brachiocephalic_trunk', ...
    'brachiocephalic_vein', 'common_carotid_artery', ...
    'subclavian_artery', 'aorta', 'superior_vena_cava', ...
    'inferior_vena_cava', 'portal_and_splenic_vein', ...
    'pulmonary_vein', 'trachea', 'lung', 'heart', ...
    'atrial_appendage', 'esophagus', 'stomach', 'duodenum', ...
    'small_bowel', 'colon', 'liver', 'gallbladder', ...
    'pancreas', 'spleen', 'thyroid_gland', 'kidney', ...
    'kidney_cyst', 'urinary_bladder', 'prostate', 'spinal_cord' ...
    };
allValues = [ ...
  0.30, ...
  0.25, ... % adrenal_glands (glandular soft tissue)
  0.40, ... % autochthon (deep back muscle)
  0.02, ... % clavicula (cortical bone)
  0.08, ... % costal_cartilages (cartilage)
  0.05, ... % ribs (cortical bone)
  0.02, ... % scapula (cortical bone)
  0.02, ... % sternum (cortical bone)
  0.02, ... % skull (cortical bone)
  0.02, ... % humerus (cortical bone)
  0.02, ... % femur (cortical bone)
  0.02, ... % vertebrae (cortical bone)
  0.02, ... % sacrum (cortical bone)
  0.40, ... % gluteus_maximus (skeletal muscle)
  0.40, ... % gluteus_minimus (skeletal muscle)
  0.40, ... % gluteus_medius (skeletal muscle)
  0.40, ... % iliopsoas (skeletal muscle)
  0.08, ... % hip (articular cartilage)
  0.70, ... % iliac_arteries (blood)
  0.70, ... % iliac_veins (blood)
  0.70, ... % brachiocephalic_trunk (blood)
  0.70, ... % brachiocephalic_veins (blood)
  0.70, ... % common_carotid_arteries (blood)
  0.70, ... % subclavian_arteries (blood)
  0.70, ... % aorta (blood)
  0.70, ... % superior_vena_cava (blood)
  0.70, ... % inferior_vena_cava (blood)
  0.70, ... % portal_and_splenic_vein (blood)
  0.70, ... % pulmonary_vein (blood)
  0.15, ... % trachea (cartilage)
  0.15, ... % lungs (inflated lung)
  0.50, ... % heart (myocardium)
  0.20, ... % atrial_appendage (myocardium)
  0.40, ... % esophagus (smooth muscle)
  0.40, ... % stomach (smooth muscle)
  0.40, ... % duodenum (smooth muscle)
  0.40, ... % small_bowel (smooth muscle)
  0.40, ... % colon (smooth muscle)
  0.20, ... % liver (parenchyma)
  0.15, ... % gallbladder (wall tissue)
  0.15, ... % pancreas (parenchyma)
  0.20, ... % spleen (parenchyma)
  0.50, ... % thyroid_gland (glandular tissue)
  0.35, ... % kidneys (parenchyma)
  1.50, ... % kidney_cysts (fluid-filled)
  1.50, ... % urinary_bladder (urine)
  0.35, ... % prostate (glandular tissue)
  0.30  ... % spinal_cord (mixed white/grey matter)
];





fullMap = containers.Map(allNames, allValues);


%--- Étape 1 : convertir ce char array en cell array de lignes
orgListCell = cellstr(orgList);
% now orgListCell est un cell array { 'adrenal_glands'; 'autochthon'; ... }

%--- Étape 2 : construire masterNames
masterNames = [{'soft_tissue'}, orgListCell.'];   % cell array 1×(1+numOrg)

%--- Exemple de boucle sans erreur
masterValues = zeros(1,numel(masterNames))';
for k = 1:numel(masterNames)
    name = masterNames{k};           % OK, c'est bien un char
    if fullMap.isKey(name)
        masterValues(k) = fullMap(name);
    else
        error('Pas de conductivité définie pour "%s".', name);
    end
end

%--- 4) On fait le mapping label→σ
sigma_real = zeros(size(sig));

% label 1 → soft_tissue
sigma_real(sig==1) = masterValues(1);

% labels 2… = organes dans orgList
nn=numel(orgListCell)+1;
for k = 2:nn
    label = k;                % convention : 2 → orgList{1}, 3 → orgList{2}, …
    sigma_real(sig==label) = masterValues(k);
end
end