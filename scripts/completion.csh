complete do_dssp "n/-tu/( ps fs ns us ms s m h)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-ssdump/f:*.dat{,.gz,.Z}/" "n/-map/f:*.map{,.gz,.Z}/" "n/-o/f:*.xpm{,.gz,.Z}/" "n/-sc/f:*.xvg{,.gz,.Z}/" "n/-a/f:*.xpm{,.gz,.Z}/" "n/-ta/f:*.xvg{,.gz,.Z}/" "n/-aa/f:*.xvg{,.gz,.Z}/" "c/-/( f s n ssdump map o sc a ta aa h X nice b e dt tu w sss)/"
complete editconf "n/-bt/( tric cubic dodecahedron octahedron)/" "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-bf/f:*.dat{,.gz,.Z}/" "c/-/( f n o bf h X nice w ndef bt box angles d c center rotate princ scale density pbc mead grasp rvdw atom legend label)/"
complete eneconv "n/-f/f:*.{edr,ene}{,.gz,.Z}/" "n/-o/f:*.{edr,ene}{,.gz,.Z}/" "c/-/( f o h X nice b e dt offset settime nosort scalefac noerror)/"
complete g_anaeig "n/-tu/( ps fs ns us ms s m h)/" "n/-v/f:*.{trr,trj}{,.gz,.Z}/" "n/-v2/f:*.{trr,trj}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-eig1/f:*.xvg{,.gz,.Z}/" "n/-eig2/f:*.xvg{,.gz,.Z}/" "n/-disp/f:*.xvg{,.gz,.Z}/" "n/-proj/f:*.xvg{,.gz,.Z}/" "n/-2d/f:*.xvg{,.gz,.Z}/" "n/-3d/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-filt/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-extr/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-over/f:*.xvg{,.gz,.Z}/" "n/-inpr/f:*.xpm{,.gz,.Z}/" "c/-/( v v2 f s n eig1 eig2 disp proj 2d 3d filt extr over inpr h X nice b e dt tu w first last skip max nframes split)/"
complete g_analyze "n/-errbar/( none stddev error 90)/" "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-f/f:*.xvg{,.gz,.Z}/" "n/-ac/f:*.xvg{,.gz,.Z}/" "n/-msd/f:*.xvg{,.gz,.Z}/" "n/-cc/f:*.xvg{,.gz,.Z}/" "n/-dist/f:*.xvg{,.gz,.Z}/" "n/-av/f:*.xvg{,.gz,.Z}/" "n/-ee/f:*.xvg{,.gz,.Z}/" "c/-/( f ac msd cc dist av ee h X nice w notime b e n d bw errbar power nosubav oneacf acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_angle "n/-type/( angle dihedral improper ryckaert-bellemans)/" "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-ov/f:*.xvg{,.gz,.Z}/" "n/-of/f:*.xvg{,.gz,.Z}/" "n/-ot/f:*.xvg{,.gz,.Z}/" "n/-oh/f:*.xvg{,.gz,.Z}/" "n/-oc/f:*.xvg{,.gz,.Z}/" "c/-/( f s n od ov of ot oh oc h X nice b e dt w type all binwidth chandler avercorr acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_bond "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-l/f:*.log{,.gz,.Z}/" "c/-/( f n o l h X nice b e dt w blen tol noaver)/"
complete g_bundle "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-ol/f:*.xvg{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-ot/f:*.xvg{,.gz,.Z}/" "n/-otr/f:*.xvg{,.gz,.Z}/" "n/-otl/f:*.xvg{,.gz,.Z}/" "n/-oa/f:*.pdb{,.gz,.Z}/" "c/-/( f s n ol od ot otr otl oa h X nice b e dt na z)/"
complete g_chi "n/-maxchi/( 0 1 2 3 4 5 6)/" "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-c/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-p/f:*.pdb{,.gz,.Z}/" "n/-ss/f:*.dat{,.gz,.Z}/" "n/-jc/f:*.xvg{,.gz,.Z}/" "n/-corr/f:*.xvg{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "c/-/( c f o p ss jc corr g h X nice b e dt w r0 phi psi omega rama viol all shift run maxchi nonormhisto ramomega bfact bmax acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_cluster "n/-tu/( ps fs ns us ms s m h)/" "n/-method/( linkage jarvis-patrick monte-carlo diagonalization gromos)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-dm/f:*.xpm{,.gz,.Z}/" "n/-o/f:*.xpm{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "n/-dist/f:*.xvg{,.gz,.Z}/" "n/-ev/f:*.xvg{,.gz,.Z}/" "n/-sz/f:*.xvg{,.gz,.Z}/" "n/-tr/f:*.xpm{,.gz,.Z}/" "n/-ntr/f:*.xvg{,.gz,.Z}/" "n/-clid/f:*.xvg{,.gz,.Z}/" "n/-cl/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "c/-/( f s n dm o g dist ev sz tr ntr clid cl h X nice b e dt tu w dista nlevels keepfree cutoff max skip av wcl nst rmsmin method binary M P seed niter kT)/"
complete g_confrms "n/-f1/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-f2/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n1/f:*.ndx{,.gz,.Z}/" "n/-n2/f:*.ndx{,.gz,.Z}/" "c/-/( f1 f2 o n1 n2 h X nice one pbc)/"
complete g_covar "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-v/f:*.{trr,trj}{,.gz,.Z}/" "n/-av/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-l/f:*.log{,.gz,.Z}/" "n/-xpm/f:*.xpm{,.gz,.Z}/" "n/-xpma/f:*.xpm{,.gz,.Z}/" "c/-/( f s n o v av l xpm xpma h X nice b e dt nofit ref mwa last)/"
complete g_density "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-ei/f:*.dat{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f n s ei o h X nice b e dt w d sl number ed count)/"
complete g_dielectric "n/-ffn/( none exp aexp exp_exp vac)/" "n/-f/f:*.xvg{,.gz,.Z}/" "n/-d/f:*.xvg{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-c/f:*.xvg{,.gz,.Z}/" "c/-/( f d o c h X nice b e dt w fft nox1 eint bfit efit tail A tau1 tau2 eps0 epsRF fix ffn nsmooth)/"
complete g_dih "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.out{,.gz,.Z}/" "c/-/( f s o h X nice b e dt w sa mult)/"
complete g_dipoles "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-enx/f:*.{edr,ene}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-e/f:*.xvg{,.gz,.Z}/" "n/-a/f:*.xvg{,.gz,.Z}/" "n/-d/f:*.xvg{,.gz,.Z}/" "n/-c/f:*.xvg{,.gz,.Z}/" "n/-g/f:*.xvg{,.gz,.Z}/" "n/-q/f:*.xvg{,.gz,.Z}/" "c/-/( enx f s n o e a d c g q h X nice b e dt w mu mumax epsilonRF skip temp avercorr gkratom acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_disre "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-ds/f:*.xvg{,.gz,.Z}/" "n/-da/f:*.xvg{,.gz,.Z}/" "n/-dn/f:*.xvg{,.gz,.Z}/" "n/-dm/f:*.xvg{,.gz,.Z}/" "n/-dr/f:*.xvg{,.gz,.Z}/" "n/-l/f:*.log{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( s f ds da dn dm dr l n h X nice b e dt w ntop)/"
complete g_dist "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f s n o h X nice b e dt dist)/"
complete g_dyndom "n/-f/f:*.pdb{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( f o n h X nice firstangle lastangle nframe maxangle trans head tail)/"
complete g_enemat "n/-f/f:*.{edr,ene}{,.gz,.Z}/" "n/-groups/f:*.dat{,.gz,.Z}/" "n/-eref/f:*.dat{,.gz,.Z}/" "n/-emat/f:*.xpm{,.gz,.Z}/" "n/-etot/f:*.xvg{,.gz,.Z}/" "c/-/( f groups eref emat etot h X nice b e dt w sum skip nomean nlevels max min nocoul coulr coul14 nolj lj14 bham nofree temp)/"
complete g_energy "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-f/f:*.{edr,ene}{,.gz,.Z}/" "n/-f2/f:*.{edr,ene}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-viol/f:*.xvg{,.gz,.Z}/" "n/-pairs/f:*.xvg{,.gz,.Z}/" "n/-corr/f:*.xvg{,.gz,.Z}/" "n/-vis/f:*.xvg{,.gz,.Z}/" "n/-ravg/f:*.xvg{,.gz,.Z}/" "c/-/( f f2 s o viol pairs corr vis ravg h X nice b e w fee fetemp zero sum dp mutot skip aver nmol ndf fluc acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_gyrate "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( f s o n h X nice b e dt w q p)/"
complete g_h2order "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-nm/f:*.ndx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f n nm s o h X nice b e dt w d sl)/"
complete g_hbond "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "n/-sel/f:*.ndx{,.gz,.Z}/" "n/-num/f:*.xvg{,.gz,.Z}/" "n/-ac/f:*.xvg{,.gz,.Z}/" "n/-dist/f:*.xvg{,.gz,.Z}/" "n/-ang/f:*.xvg{,.gz,.Z}/" "n/-hx/f:*.xvg{,.gz,.Z}/" "n/-hbn/f:*.ndx{,.gz,.Z}/" "n/-hbm/f:*.xpm{,.gz,.Z}/" "n/-da/f:*.xvg{,.gz,.Z}/" "c/-/( f s n g sel num ac dist ang hx hbn hbm da h X nice b e dt ins a r abin rbin nonitacc shell)/"
complete g_helix "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-to/f:*.g87{,.gz,.Z}/" "n/-cz/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-co/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "c/-/( s n f to cz co h X nice b e dt w r0 q noF db ev ahxstart ahxend)/"
complete g_lie "n/-f/f:*.{edr,ene}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f o h X nice b e dt w Elj Eqq Clj Cqq ligand)/"
complete g_mdmat "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-mean/f:*.xpm{,.gz,.Z}/" "n/-frames/f:*.xpm{,.gz,.Z}/" "n/-no/f:*.xvg{,.gz,.Z}/" "c/-/( f s n mean frames no h X nice b e dt t nlevels)/"
complete g_mindist "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-on/f:*.xvg{,.gz,.Z}/" "n/-o/f:*.out{,.gz,.Z}/" "c/-/( f s n od on o h X nice b e dt w matrix d pi)/"
complete g_morph "n/-f1/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-f2/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-or/f:*.xvg{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( f1 f2 o or n h X nice w ninterm first last nofit)/"
complete g_msd "n/-tu/( ps fs ns us ms s m h)/" "n/-type/( no x y z)/" "n/-lateral/( no x y z)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-mol/f:*.xvg{,.gz,.Z}/" "c/-/( f s n o mol h X nice b e dt tu w type lateral ngroup nomw trestart beginfit endfit)/"
complete g_nmeig "n/-f/f:*.mtx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-v/f:*.{trr,trj}{,.gz,.Z}/" "c/-/( f s o v h X nice nom first last)/"
complete g_nmens "n/-v/f:*.{trr,trj}{,.gz,.Z}/" "n/-e/f:*.xvg{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "c/-/( v e s n o h X nice temp seed num first last)/"
complete g_order "n/-d/( z x y)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-os/f:*.xvg{,.gz,.Z}/" "c/-/( f n s o od os h X nice b e dt w d sl szonly unsat)/"
complete g_potential "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-oc/f:*.xvg{,.gz,.Z}/" "n/-of/f:*.xvg{,.gz,.Z}/" "c/-/( f n s o oc of h X nice b e dt w d sl cb ce tz spherical)/"
complete g_rama "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f s o h X nice b e dt w)/"
complete g_rdf "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-sq/f:*.xvg{,.gz,.Z}/" "n/-cn/f:*.xvg{,.gz,.Z}/" "n/-hq/f:*.xvg{,.gz,.Z}/" "n/-image/f:*.xpm{,.gz,.Z}/" "c/-/( f s n o sq cn hq image h X nice b e dt w bin com cut fade grid nlevel wave)/"
complete g_rms "n/-tu/( ps fs ns us ms s m h)/" "n/-what/( rmsd rho rhosc)/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-f2/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-mir/f:*.xvg{,.gz,.Z}/" "n/-a/f:*.xvg{,.gz,.Z}/" "n/-dist/f:*.xvg{,.gz,.Z}/" "n/-m/f:*.xpm{,.gz,.Z}/" "n/-bin/f:*.dat{,.gz,.Z}/" "n/-bm/f:*.xpm{,.gz,.Z}/" "c/-/( s f f2 n o mir a dist m bin bm h X nice b e dt tu w what nopbc nofit prev split skip skip2 max min bmax bmin nlevels)/"
complete g_rmsdist "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-equiv/f:*.dat{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-rms/f:*.xpm{,.gz,.Z}/" "n/-scl/f:*.xpm{,.gz,.Z}/" "n/-mean/f:*.xpm{,.gz,.Z}/" "n/-nmr3/f:*.xpm{,.gz,.Z}/" "n/-nmr6/f:*.xpm{,.gz,.Z}/" "n/-noe/f:*.dat{,.gz,.Z}/" "c/-/( f s n equiv o rms scl mean nmr3 nmr6 noe h X nice b e dt w nlevels max nosumh)/"
complete g_rmsf "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-q/f:*.pdb{,.gz,.Z}/" "n/-oq/f:*.pdb{,.gz,.Z}/" "n/-ox/f:*.pdb{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-oc/f:*.xvg{,.gz,.Z}/" "n/-dir/f:*.log{,.gz,.Z}/" "c/-/( f s n q oq ox o od oc dir h X nice b e dt w res aniso)/"
complete g_rotacf "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f s n o h X nice b e dt w d noaver acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete g_saltbr "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "c/-/( f s h X nice b e dt t sep)/"
complete g_sas "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-r/f:*.xvg{,.gz,.Z}/" "n/-q/f:*.pdb{,.gz,.Z}/" "n/-ao/f:*.xvg{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-i/f:*.itp{,.gz,.Z}/" "c/-/( f s o r q ao n i h X nice b e dt w solsize ndots qmax minarea skip noprot)/"
complete g_sgangle "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-oa/f:*.xvg{,.gz,.Z}/" "n/-od/f:*.xvg{,.gz,.Z}/" "n/-od1/f:*.xvg{,.gz,.Z}/" "n/-od2/f:*.xvg{,.gz,.Z}/" "c/-/( f n s oa od od1 od2 h X nice b e dt w)/"
complete g_sorient "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-no/f:*.xvg{,.gz,.Z}/" "n/-ro/f:*.xvg{,.gz,.Z}/" "n/-co/f:*.xvg{,.gz,.Z}/" "c/-/( f s n o no ro co h X nice b e dt w com rmin rmax nbin)/"
complete g_tcaf "n/-f/f:*.{trr,trj}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-ot/f:*.xvg{,.gz,.Z}/" "n/-oa/f:*.xvg{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "n/-of/f:*.xvg{,.gz,.Z}/" "n/-oc/f:*.xvg{,.gz,.Z}/" "n/-ov/f:*.xvg{,.gz,.Z}/" "c/-/( f s n ot oa o of oc ov h X nice b e dt w mol k34 wt)/"
complete g_traj "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-ox/f:*.xvg{,.gz,.Z}/" "n/-ov/f:*.xvg{,.gz,.Z}/" "n/-of/f:*.xvg{,.gz,.Z}/" "n/-ob/f:*.xvg{,.gz,.Z}/" "n/-ot/f:*.xvg{,.gz,.Z}/" "n/-ekr/f:*.xvg{,.gz,.Z}/" "c/-/( f s n ox ov of ob ot ekr h X nice b e dt w com mol nojump nox noy noz len)/"
complete g_velacc "n/-P/( 0 1 2 3)/" "n/-fitfn/( none exp aexp exp_exp vac)/" "n/-f/f:*.{trr,trj}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.xvg{,.gz,.Z}/" "c/-/( f s n o h X nice b e dt w mol acflen nonormalize P fitfn ncskip beginfit endfit)/"
complete genbox "n/-cp/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-cs/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-ci/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-p/f:*.top{,.gz,.Z}/" "c/-/( cp cs ci o p h X nice box nmol try seed vdwd shell)/"
complete genconf "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-trj/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "c/-/( f o trj h X nice nbox dist seed rot shuffle sort block nmolat maxrot renumber)/"
complete genion "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "n/-pot/f:*.pdb{,.gz,.Z}/" "c/-/( s n o g pot h X nice np pname pq nn nname nq rmin random seed)/"
complete genpr "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.itp{,.gz,.Z}/" "c/-/( f n o h X nice fc)/"
complete gmxcheck "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-f2/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s1/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-s2/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-c/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-e/f:*.{edr,ene}{,.gz,.Z}/" "n/-e2/f:*.{edr,ene}{,.gz,.Z}/" "c/-/( f f2 s1 s2 c e e2 h X nice vdwfac bonlo bonhi tol)/"
complete gmxdump "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-e/f:*.{edr,ene}{,.gz,.Z}/" "c/-/( s f e h X nice nonr)/"
complete grompp "n/-f/f:*.mdp{,.gz,.Z}/" "n/-po/f:*.mdp{,.gz,.Z}/" "n/-c/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-r/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-deshuf/f:*.ndx{,.gz,.Z}/" "n/-p/f:*.top{,.gz,.Z}/" "n/-pp/f:*.top{,.gz,.Z}/" "n/-o/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-t/f:*.{trr,trj}{,.gz,.Z}/" "c/-/( f po c r n deshuf p pp o t h X nice nov time np shuffle sort normdumbds load maxwarn check14)/"
complete highway "n/-f/f:*.dat{,.gz,.Z}/" "n/-a/f:*.dat{,.gz,.Z}/" "c/-/( f a h X nice b e dt)/"
complete make_ndx "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.ndx{,.gz,.Z}/" "c/-/( f n o h X nice)/"
complete mdrun "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{trr,trj}{,.gz,.Z}/" "n/-x/f:*.xtc{,.gz,.Z}/" "n/-c/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-e/f:*.{edr,ene}{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "n/-dgdl/f:*.xvg{,.gz,.Z}/" "n/-table/f:*.xvg{,.gz,.Z}/" "n/-rerun/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-ei/f:*.edi{,.gz,.Z}/" "n/-eo/f:*.edo{,.gz,.Z}/" "n/-pi/f:*.ppa{,.gz,.Z}/" "n/-po/f:*.ppa{,.gz,.Z}/" "n/-pd/f:*.pdo{,.gz,.Z}/" "n/-pn/f:*.ndx{,.gz,.Z}/" "c/-/( s o x c e g dgdl table rerun ei eo pi po pd pn h X nice deffnm np v nocompact)/"
complete mk_angndx "n/-type/( angle g96-angle dihedral improper ryckaert-bellemans phi-psi)/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( s n h X nice type)/"
complete ngmx "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( f s n h X nice b e dt)/"
complete nmrun "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-m/f:*.mtx{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "c/-/( s m g h X nice np v nocompact)/"
complete pdb2gmx "n/-dummy/( none hydrogens aromatics)/" "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-p/f:*.top{,.gz,.Z}/" "n/-i/f:*.itp{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-q/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "c/-/( f o p i n q h X nice merge inter ss ter lys asp glu his angle dist una nosort H14 ignh alldih dummy heavyh deuterate)/"
complete protonate "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "c/-/( s f n o h X nice b e dt)/"
complete tpbconv "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-f/f:*.{trr,trj}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "c/-/( s f n o h X nice time extend until nounconstrained)/"
complete trjcat "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "c/-/( o n h X nice b e dt prec novel settime nosort)/"
complete trjconv "n/-tu/( ps fs ns us ms s m h)/" "n/-pbc/( none whole inbox nojump)/" "n/-ur/( rect tric compact)/" "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-fr/f:*.ndx{,.gz,.Z}/" "c/-/( f o s n fr h X nice b e tu w skip dt dump t0 timestep pbc ur center box shift fit pfit ndec novel force trunc exec app split sep)/"
complete trjorder "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa,gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-n/f:*.ndx{,.gz,.Z}/" "n/-o/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "c/-/( f s n o h X nice b e dt na da)/"
complete wheel "n/-f/f:*.dat{,.gz,.Z}/" "n/-o/f:*.eps{,.gz,.Z}/" "c/-/( f o h X nice r0 rot0 T nonn)/"
complete x2top "n/-f/f:*.{gro,g96,pdb,brk,ent,tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.top{,.gz,.Z}/" "n/-r/f:*.rtp{,.gz,.Z}/" "c/-/( f o r h X nice kb kt kp nexcl H14 alldih noround nopairs name)/"
complete xmdrun "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "n/-o/f:*.{trr,trj}{,.gz,.Z}/" "n/-x/f:*.xtc{,.gz,.Z}/" "n/-c/f:*.{gro,g96,pdb,brk,ent}{,.gz,.Z}/" "n/-e/f:*.{edr,ene}{,.gz,.Z}/" "n/-g/f:*.log{,.gz,.Z}/" "n/-dgdl/f:*.xvg{,.gz,.Z}/" "n/-table/f:*.xvg{,.gz,.Z}/" "n/-rerun/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-ei/f:*.edi{,.gz,.Z}/" "n/-eo/f:*.edo{,.gz,.Z}/" "n/-j/f:*.gct{,.gz,.Z}/" "n/-jo/f:*.gct{,.gz,.Z}/" "n/-ffout/f:*.xvg{,.gz,.Z}/" "n/-devout/f:*.xvg{,.gz,.Z}/" "n/-runav/f:*.xvg{,.gz,.Z}/" "n/-pi/f:*.ppa{,.gz,.Z}/" "n/-po/f:*.ppa{,.gz,.Z}/" "n/-pd/f:*.pdo{,.gz,.Z}/" "n/-pn/f:*.ndx{,.gz,.Z}/" "c/-/( s o x c e g dgdl table rerun ei eo j jo ffout devout runav pi po pd pn h X nice deffnm np v nocompact multi glas ionize)/"
complete xpm2ps "n/-title/( top once ylabel none)/" "n/-legend/( both first second none)/" "n/-diag/( first second none)/" "n/-combine/( halves add sub mult div)/" "n/-rainbow/( no blue red)/" "n/-f/f:*.xpm{,.gz,.Z}/" "n/-f2/f:*.xpm{,.gz,.Z}/" "n/-di/f:*.m2p{,.gz,.Z}/" "n/-do/f:*.m2p{,.gz,.Z}/" "n/-o/f:*.eps{,.gz,.Z}/" "n/-xpm/f:*.xpm{,.gz,.Z}/" "c/-/( f f2 di do o xpm h X nice w noframe title yonce legend diag combine bx by rainbow gradient skip zeroline)/"
complete xrama "n/-f/f:*.{xtc,trr,trj,gro,g96,pdb,g87}{,.gz,.Z}/" "n/-s/f:*.{tpr,tpb,tpa}{,.gz,.Z}/" "c/-/( f s h X nice b e dt)/"
