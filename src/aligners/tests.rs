use itertools::Itertools;
use rand::{seq::IteratorRandom, thread_rng, Rng};

use super::{
    cigar::test::verify_cigar,
    diagonal_transition::{DiagonalTransition, GapCostHeuristic},
    nw::NW,
    Aligner, Seq,
};
use crate::{
    generate::{setup_sequences_with_seed_and_model, ErrorModel},
    heuristic::NoCost,
    prelude::{to_string, AffineCost, AffineLayerCosts, AffineLayerType},
    visualizer::NoVisualizer,
};

fn test_sequences() -> impl Iterator<Item = (((usize, f32), ErrorModel), u64)> {
    let rng = &mut thread_rng();
    // Each run picks a random sample of the numbers below to speed things up.
    let ns = [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50,
        60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
        250, 260, 270, 280, 290, 300, 500,
    ];
    let ns = ns.into_iter().choose_multiple(rng, ns.len() / 2);
    let es = [
        0.0f32, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.0,
    ];
    let es = es.into_iter().choose_multiple(rng, es.len() / 2);
    let models = [
        ErrorModel::Uniform,
        ErrorModel::NoisyInsert,
        ErrorModel::NoisyDelete,
        ErrorModel::DoubleMutatedRepeat,
    ];
    // Run each test on a new random seed for increased coverage over time.
    let seeds = [rng.gen_range(0..u64::MAX)];
    ns.into_iter()
        .cartesian_product(es)
        .cartesian_product(models)
        .cartesian_product(seeds)
}

fn test_aligner_on_input<const N: usize, A: Aligner>(
    a: Seq,
    b: Seq,
    aligner: &mut impl Aligner,
    viz_aligner: &mut Option<&mut dyn FnMut(&[u8], &[u8]) -> A>,
    test_path: bool,
    cm: &AffineCost<N>,
    params: &str,
) {
    // Set to true for local debugging.
    const D: bool = false;

    // useful in case of panics inside the alignment code.
    eprintln!("{params}");
    if D {
        eprintln!("a {}\nb {}", to_string(a), to_string(b));
    }
    let mut nw = NW::new(cm.clone(), false, false);
    let nw_cost = nw.cost(a, b);
    let cost = aligner.cost(a, b);
    // Rerun the alignment with the visualizer enabled.
    if D && nw_cost != cost && let Some(viz_aligner) = viz_aligner {
        eprintln!("{params}\na: {}\nb: {}\nnw_cost: {nw_cost}\ntest_cost: {cost}\n", to_string(a), to_string(b));
        viz_aligner(a, b).align(a, b);
    }
    // Test the cost reported by all aligners.
    assert_eq!(
        nw_cost,
        cost,
        "\n{params}\nlet a = \"{}\".as_bytes();\nlet b = \"{}\".as_bytes();\nNW cigar: {}\nAligner\n{aligner:?}",
        to_string(&a),
        to_string(&b),
        nw.align(a, b).1.to_string()
    );
    if test_path {
        let (cost, cigar) = aligner.align(a, b);
        if cost != nw_cost {
            eprintln!("\n================= TEST CIGAR ======================\n");
            eprintln!(
                "{params}\nlet a = \"{}\".as_bytes();\nlet b = \"{}\".as_bytes();\ncigar: {}\nnwcig: {}",
                to_string(a),
                to_string(b),
                cigar.to_string(),
                nw.align(a, b).1.to_string()
            );
        }
        assert_eq!(cost, nw_cost);
        verify_cigar(cm, a, b, &cigar);
    }
}

/// Test that:
/// - the aligner gives the same cost as NW, both for `cost` and `align` members.
/// - the `Cigar` is valid and of the correct cost.
fn test_aligner_on_cost_model_with_viz<const N: usize, A: Aligner>(
    cm: AffineCost<N>,
    mut aligner: impl Aligner,
    mut viz_aligner: Option<&mut dyn FnMut(Seq, Seq) -> A>,
    test_path: bool,
) {
    for (((n, e), error_model), seed) in test_sequences() {
        let (ref a, ref b) = setup_sequences_with_seed_and_model(seed, n, e, error_model);
        test_aligner_on_input(
            a,
            b,
            &mut aligner,
            &mut viz_aligner,
            test_path,
            &cm,
            &format!("seed {seed} n {n} e {e} error_model {error_model:?}"),
        );
    }
}

fn test_aligner_on_cost_model<const N: usize, A: Aligner>(
    cm: AffineCost<N>,
    aligner: A,
    test_path: bool,
) {
    let a: Option<&mut dyn FnMut(Seq, Seq) -> A> = None;
    test_aligner_on_cost_model_with_viz(cm, aligner, a, test_path);
}

mod triple_accel {
    use crate::{aligners::triple_accel::TripleAccel, cost_model::CostModel::Levenshtein};

    use super::*;

    #[test]
    fn unit_cost() {
        let cm = AffineCost::new_unit();
        test_aligner_on_cost_model(cm.clone(), TripleAccel::new(false, Levenshtein), false);
    }

    #[test]
    fn unit_cost_exp() {
        let cm = AffineCost::new_unit();
        test_aligner_on_cost_model(cm.clone(), TripleAccel::new(false, Levenshtein), false);
    }
}

mod astar {
    use std::marker::PhantomData;

    use crate::{
        aligners::astar::AstarPA,
        cost_model::LinearCost,
        heuristic::{Heuristic, Pruning, CSH, SH},
        matches::MatchConfig,
        prelude::{BruteForceContour, HintContours},
    };

    use super::*;

    fn test_heuristic<H: Heuristic>(h: H, dt: bool) {
        // Greedy matching doesn't really matter much.
        // To speed up tests, we choose it randomly.
        test_aligner_on_cost_model_with_viz(
            LinearCost::new_unit(),
            AstarPA {
                dt,
                h,
                v: NoVisualizer,
            },
            Some(&mut |_a, _b| AstarPA {
                dt,
                h,
                v: {
                    #[cfg(feature = "vis")]
                    {
                        use crate::visualizer::{Config, VisualizerStyle};
                        Config::new(VisualizerStyle::Test)
                    }
                    #[cfg(not(feature = "vis"))]
                    {
                        NoVisualizer
                    }
                },
            }),
            true,
        );
    }

    macro_rules! make_test {
        // h is a function (exact: bool, pruning: bool) -> Heuristic.
        ($name:ident, $h:expr) => {
            mod $name {
                use super::*;
                // large k variants with mostly linear matches
                #[test]
                fn exact_noprune() {
                    super::test_heuristic($h(true, false, false), false);
                }
                #[test]
                fn exact_prune() {
                    super::test_heuristic($h(true, true, false), false);
                }
                #[test]
                fn inexact_noprune() {
                    super::test_heuristic($h(false, false, false), false);
                }
                #[test]
                fn inexact_prune() {
                    super::test_heuristic($h(false, true, false), false);
                }
                #[test]
                fn exact_noprune_dt() {
                    super::test_heuristic($h(true, false, false), true);
                }
                #[test]
                fn exact_prune_dt() {
                    super::test_heuristic($h(true, true, false), true);
                }
                #[test]
                fn inexact_noprune_dt() {
                    super::test_heuristic($h(false, false, false), true);
                }
                #[test]
                fn inexact_prune_dt() {
                    super::test_heuristic($h(false, true, false), true);
                }

                // small k variants with many matches, to stress the contours
                #[test]
                fn exact_noprune_smallk() {
                    super::test_heuristic($h(true, false, true), false);
                }
                #[test]
                fn exact_prune_smallk() {
                    super::test_heuristic($h(true, true, true), false);
                }
                #[test]
                fn inexact_noprune_smallk() {
                    super::test_heuristic($h(false, false, true), false);
                }
                #[test]
                fn inexact_prune_smallk() {
                    super::test_heuristic($h(false, true, true), false);
                }
                #[test]
                fn exact_noprune_dt_smallk() {
                    super::test_heuristic($h(true, false, true), true);
                }
                #[test]
                fn exact_prune_dt_smallk() {
                    super::test_heuristic($h(true, true, true), true);
                }
                #[test]
                fn inexact_noprune_dt_smallk() {
                    super::test_heuristic($h(false, false, true), true);
                }
                #[test]
                fn inexact_prune_dt_smallk() {
                    super::test_heuristic($h(false, true, true), true);
                }
            }
        };
    }

    mod dijkstra {
        use crate::heuristic::NoCost;

        #[test]
        fn exact_noprune() {
            super::test_heuristic(NoCost, false);
        }
        #[test]
        fn exact_noprune_dt() {
            super::test_heuristic(NoCost, true);
        }
    }

    fn match_config(exact: bool, small_k: bool) -> MatchConfig {
        match (exact, small_k) {
            (true, false) => MatchConfig::exact(5),
            (true, true) => MatchConfig::exact(2),
            (false, false) => MatchConfig::inexact(9),
            (false, true) => MatchConfig::inexact(3),
        }
    }

    // normal k with few matches
    make_test!(sh, |exact, prune, small_k| SH {
        match_config: match_config(exact, small_k),
        pruning: Pruning::new(prune)
    });
    make_test!(csh, |exact, prune, small_k| CSH {
        match_config: match_config(exact, small_k),
        pruning: Pruning::new(prune),
        use_gap_cost: false,
        c: PhantomData::<HintContours<BruteForceContour>>,
    });
    make_test!(gch, |exact, prune, small_k| CSH {
        match_config: match_config(exact, small_k),
        pruning: Pruning::new(prune),
        use_gap_cost: true,
        c: PhantomData::<HintContours<BruteForceContour>>,
    });
}

mod edlib {
    use crate::{aligners::edlib::Edlib, cost_model::LinearCost};

    use super::*;

    #[test]
    fn unit_cost() {
        // sub=indel=1
        test_aligner_on_cost_model(LinearCost::new_unit(), Edlib, false);
    }
}

mod biwfa {
    use crate::aligners::wfa::WFA;

    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(cm.clone(), WFA { cm }, false);
    }

    #[test]
    fn unit_cost() {
        // sub=indel=1
        test(AffineCost::new_unit());
    }

    #[test]
    fn lcs_cost() {
        // sub=infinity, indel=1
        test(AffineCost::new_lcs());
    }

    #[test]
    fn linear_cost() {
        // sub=1, indel=2
        test(AffineCost::new_linear(1, 2));
    }

    #[test]
    fn linear_cost_3() {
        // sub=1, indel=3
        test(AffineCost::new_linear(1, 3));
    }

    #[test]
    fn affine_cost() {
        // sub=1
        // open=2, extend=1
        test(AffineCost::new_affine(1, 2, 1));
    }

    #[test]
    fn double_affine_cost() {
        // sub=1
        // Gap cost is min(4+2*l, 10+1*l).
        test(AffineCost::new_double_affine(1, 4, 2, 10, 1));
    }
}

macro_rules! test_functions_macro {
    () => {
        #[test]
        fn lcs_cost() {
            // sub=infinity, indel=1
            test(AffineCost::new_lcs());
        }

        #[test]
        fn unit_cost() {
            // sub=indel=1
            test(AffineCost::new_unit());
        }

        #[test]
        fn linear_cost() {
            // sub=1, indel=2
            test(AffineCost::new_linear(1, 2));
        }

        #[test]
        fn linear_cost_3() {
            // sub=1, indel=3
            test(AffineCost::new_linear(1, 3));
        }

        #[test]
        fn linear_asymmetric_cost() {
            // sub=1, insert=2, deletion=3
            test(AffineCost::new_linear_asymmetric(1, 2, 3));
        }

        #[test]
        fn affine_cost() {
            // sub=1
            // open=2, extend=1
            test(AffineCost::new_affine(1, 2, 1));
        }

        #[test]
        fn linear_affine_cost() {
            // sub=1, indel=3
            // open=2, extend=1
            test(AffineCost::new_linear_affine(1, 3, 2, 1));
        }

        #[ignore = "broken -- fix in the future"]
        #[test]
        fn double_affine_cost() {
            // sub=1
            // Gap cost is min(4+2*l, 10+1*l).
            test(AffineCost::new_double_affine(1, 4, 2, 10, 1));
        }

        #[test]
        fn asymmetric_affine_cost() {
            // sub=1
            // insert: open=2, extend=2
            // deletion: open=3, extend=1
            test(AffineCost::new_affine_asymmetric(1, 2, 2, 3, 1));
        }

        #[test]
        fn ins_asymmetric_affine_cost() {
            test(AffineCost::new(
                Some(1),
                Some(1),
                None,
                [AffineLayerCosts {
                    affine_type: AffineLayerType::DeleteLayer,
                    open: 2,
                    extend: 2,
                }],
            ));
        }

        #[test]
        fn del_asymmetric_affine_cost() {
            test(AffineCost::new(
                Some(1),
                None,
                Some(1),
                [AffineLayerCosts {
                    affine_type: AffineLayerType::InsertLayer,
                    open: 2,
                    extend: 2,
                }],
            ));
        }

        #[ignore = "homopolmer"]
        #[test]
        fn ins_homopolymer_cost() {
            test(AffineCost::new(
                Some(2),
                None,
                Some(3),
                [AffineLayerCosts {
                    affine_type: AffineLayerType::HomoPolymerInsert,
                    open: 2,
                    extend: 2,
                }],
            ));
        }

        #[ignore = "homopolymer"]
        #[test]
        fn del_homopolymer_cost() {
            test(AffineCost::new(
                Some(2),
                Some(3),
                None,
                [AffineLayerCosts {
                    affine_type: AffineLayerType::HomoPolymerDelete,
                    open: 2,
                    extend: 2,
                }],
            ));
        }

        #[ignore = "homopolymer"]
        #[test]
        fn indel_homopolymer_cost() {
            test(AffineCost::new(
                Some(2),
                None,
                None,
                [
                    AffineLayerCosts {
                        affine_type: AffineLayerType::HomoPolymerInsert,
                        open: 3,
                        extend: 1,
                    },
                    AffineLayerCosts {
                        affine_type: AffineLayerType::HomoPolymerDelete,
                        open: 3,
                        extend: 1,
                    },
                ],
            ));
        }

        #[ignore = "homopolymer"]
        #[test]
        fn indel_homopolymer_plus_affine_cost() {
            test(AffineCost::new(
                Some(2),
                None,
                None,
                [
                    AffineLayerCosts {
                        affine_type: AffineLayerType::InsertLayer,
                        open: 2,
                        extend: 2,
                    },
                    AffineLayerCosts {
                        affine_type: AffineLayerType::DeleteLayer,
                        open: 2,
                        extend: 2,
                    },
                    AffineLayerCosts {
                        affine_type: AffineLayerType::HomoPolymerInsert,
                        open: 3,
                        extend: 1,
                    },
                    AffineLayerCosts {
                        affine_type: AffineLayerType::HomoPolymerDelete,
                        open: 3,
                        extend: 1,
                    },
                ],
            ));
        }
    };
}

// TODO: Replace the duplication below by macros.
mod nw {

    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(cm.clone(), NW::new(cm, false, false), true);
    }

    test_functions_macro!();
}

mod exp_band_simple {
    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(cm.clone(), NW::new(cm.clone(), false, false), true);
    }

    test_functions_macro!();
}

mod exp_band_gap_heuristic {
    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(cm.clone(), NW::new(cm.clone(), true, true), true);
    }

    test_functions_macro!();
}

mod diagonal_transition_simple {
    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(
            cm.clone(),
            DiagonalTransition::new(cm, GapCostHeuristic::Disable, NoCost, false, NoVisualizer),
            true,
        );
    }

    test_functions_macro!();
}

mod diagonal_transition_gap_heuristic {
    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(
            cm.clone(),
            DiagonalTransition::new(cm, GapCostHeuristic::Enable, NoCost, false, NoVisualizer),
            true,
        );
    }

    test_functions_macro!();
}

// FIXME: Enable diagonal transition + divide & conquer tests once they are
// actually passing. For now, affine cost is not working yet.
mod diagonal_transition_dc {
    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(
            cm.clone(),
            DiagonalTransition::new(cm, GapCostHeuristic::Disable, NoCost, true, NoVisualizer),
            true,
        );
    }

    test_functions_macro!();
}

mod nw_sh {

    use crate::{
        heuristic::{Pruning, SH},
        matches::MatchConfig,
    };

    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(
            cm.clone(),
            NW {
                cm,
                use_gap_cost_heuristic: false,
                exponential_search: true,
                local_doubling: false,
                h: SH {
                    match_config: MatchConfig::exact(5),
                    pruning: Pruning::default(),
                },
                v: NoVisualizer,
            },
            // test `align` as well?
            true,
        );
    }

    #[ignore = "broken; fix in the future"]
    #[test]
    fn unit_cost() {
        // sub=indel=1
        test(AffineCost::new_unit());
    }
}

mod diagonal_transition_sh {
    use crate::{
        heuristic::{Pruning, SH},
        matches::MatchConfig,
    };

    use super::*;

    fn test<const N: usize>(cm: AffineCost<N>) {
        test_aligner_on_cost_model(
            cm.clone(),
            DiagonalTransition::new(
                cm,
                GapCostHeuristic::Disable,
                SH {
                    match_config: MatchConfig::exact(5),
                    pruning: Pruning::default(),
                },
                false,
                NoVisualizer,
            ),
            false,
        );
    }

    #[test]
    fn unit_cost() {
        // sub=indel=1
        test(AffineCost::new_unit());
    }
}

mod homopolymer {
    use crate::{
        aligners::{cigar::test::verify_cigar, nw::NW, Aligner},
        cost_model::AffineLayerType::{DeleteLayer, InsertLayer},
        heuristic::NoCost,
        prelude::{
            AffineCost, AffineLayerCosts,
            AffineLayerType::{HomoPolymerDelete, HomoPolymerInsert},
        },
        visualizer::NoVisualizer,
    };

    #[ignore = "homopolymer"]
    #[test]
    fn homo_polymer() {
        let cm = AffineCost::new(
            Some(1),
            Some(10),
            Some(10),
            [
                AffineLayerCosts {
                    affine_type: HomoPolymerInsert,
                    open: 1,
                    extend: 1,
                },
                AffineLayerCosts {
                    affine_type: HomoPolymerDelete,
                    open: 1,
                    extend: 1,
                },
            ],
        );
        let mut nw = NW {
            cm: cm.clone(),
            use_gap_cost_heuristic: false,
            exponential_search: false,
            local_doubling: false,
            h: NoCost,
            v: NoVisualizer,
        };
        assert_eq!(nw.cost(b"ABC", b"AC"), 2);
        assert_eq!(nw.cost(b"ABC", b""), 6);
        assert_eq!(nw.cost(b"ABBBC", b"AC"), 4);
        assert_eq!(nw.cost(b"ABCABCABC", b"BBBBBBBBB"), 6);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"ABCABCABC"), 6);
        assert_eq!(nw.cost(b"", b"CCCC"), 5);
        assert_eq!(nw.cost(b"", b"ABC"), 6);
        assert_eq!(nw.cost(b"ABBB", b"CBBA"), 2);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"CCCCCCC"), 10);
        assert_eq!(nw.cost(b"AAAAAAAAA", b""), 10);
    }

    #[ignore = "homopolymer"]
    #[test]
    fn homo_polymer_plus_affine_and_cigar() {
        let cm = AffineCost::new(
            Some(1),
            Some(10),
            Some(10),
            [
                AffineLayerCosts {
                    affine_type: InsertLayer,
                    open: 2,
                    extend: 2,
                },
                AffineLayerCosts {
                    affine_type: DeleteLayer,
                    open: 2,
                    extend: 2,
                },
                AffineLayerCosts {
                    affine_type: HomoPolymerInsert,
                    open: 3,
                    extend: 1,
                },
                AffineLayerCosts {
                    affine_type: HomoPolymerDelete,
                    open: 3,
                    extend: 1,
                },
            ],
        );
        let mut nw = NW {
            cm: cm.clone(),
            use_gap_cost_heuristic: false,
            exponential_search: false,
            local_doubling: false,
            v: NoVisualizer,
            h: NoCost,
        };
        assert_eq!(nw.cost(b"ABC", b"AC"), 4);
        assert_eq!(nw.cost(b"ABC", b""), 8);
        assert_eq!(nw.cost(b"ABBBC", b"AC"), 6);
        assert_eq!(nw.cost(b"ABCABCABC", b"BBBBBBBBB"), 6);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"ABCABCABC"), 6);
        assert_eq!(nw.align(b"", b"CCCC").0, 7);
        assert_eq!(nw.cost(b"", b"ABC"), 8);
        assert_eq!(nw.cost(b"ABBB", b"CBBA"), 2);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"CCCCCCC"), 12);
        assert_eq!(nw.cost(b"AAAAAAAAA", b""), 12);
        let a = b"ABC";
        let b = b"AC";
        let cigar = nw.align(a, b).1;
        verify_cigar(&cm, a, b, &cigar);

        assert_eq!(nw.cost(b"ABC", b""), 8);
        assert_eq!(nw.cost(b"ABBBC", b"AC"), 6);
        assert_eq!(nw.cost(b"ABCABCABC", b"BBBBBBBBB"), 6);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"ABCABCABC"), 6);
        assert_eq!(nw.align(b"", b"CCCC").0, 7);
        assert_eq!(nw.cost(b"", b"ABC"), 8);
        assert_eq!(nw.cost(b"ABBB", b"CBBA"), 2);
        assert_eq!(nw.cost(b"BBBBBBBBB", b"CCCCCCC"), 12);
        assert_eq!(nw.cost(b"AAAAAAAAA", b""), 12);
    }
}

// Interesting csae:
// sub: 1
// indel: 3
// G CA A TCGGG
// A CA   TCGGG
// will be found with cost 5=3+2 before finding the cost 4 path, which requires iterating up to s=4+3.
