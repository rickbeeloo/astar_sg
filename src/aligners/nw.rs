use itertools::izip;

use super::layer::Layers;
use super::NoVisualizer;
use super::{Aligner, Visualizer};
use crate::cost_model::*;
use crate::prelude::{Pos, Sequence, I};
use std::cmp::min;
use std::iter::zip;

pub type PATH = Vec<(usize, usize)>;
pub struct NW<CostModel> {
    pub cm: CostModel,
}

// TODO: Instead use saturating add everywhere?
const INF: Cost = Cost::MAX / 2;

/// The base vector M, and one vector per affine layer.
/// TODO: Possibly switch to a Vec<Layer> instead.
type NWLayers<const N: usize> = Layers<N, Vec<Cost>>;

impl<const N: usize> NW<AffineCost<N>> {
    fn track_path(&self, layers: &mut Vec<NWLayers<N>>, a: &Sequence, b: &Sequence) -> PATH {
        let mut path: PATH = vec![(layers.len(), layers[0].m.len())];
        let mut x = layers.len();
        let mut y = layers[0].m.len();
        let mut t: isize = -1;
        let mut save = |x: &usize, y: &usize| path.push((*x, *y));

        while x + y > 0 {
            let mut found = false;
            if t == -1 {
                if x * y > 0 {
                    //x*y >0 == (x > 0 && y > 0)
                    if a[x - 1] == b[y - 1] && layers[x].m[y] == layers[x - 1].m[y - 1] {
                        //match
                        x -= 1;
                        y -= 1;
                        save(&x, &y);
                        found = true;
                    }
                    if !found {
                        if let Some(sub) = self.cm.sub {
                            if layers[x].m[y] == layers[x - 1].m[y - 1] + sub {
                                //mismatch
                                x -= 1;
                                y -= 1;
                                save(&x, &y);
                                found = true;
                            }
                        }
                    }
                }
                if x > 0 && !found {
                    if let Some(ins) = self.cm.ins {
                        if layers[x].m[y] == layers[x - 1].m[y] + ins {
                            //insertion
                            x -= 1;
                            save(&x, &y);
                            found = true;
                        }
                    }
                }
                if y > 0 && !found {
                    if let Some(del) = self.cm.del {
                        if layers[x].m[y] == layers[x].m[y - 1] + del {
                            //deletion
                            y -= 1;
                            save(&x, &y);
                            found = true;
                        }
                    }
                }
                //affine layers check
                if !found {
                    let mut i = 0;
                    for affine_layer in &mut layers[x].affine {
                        if layers[x].m[y] == affine_layer[y] {
                            t = i;
                            break;
                        }
                        i += 1;
                    }
                }
            } else {
                //Attention! Loop below covers only cases when x > 0
                if x > 0 {
                    for (cm, prev_layer, next_layer) in izip!(
                        &self.cm.affine,
                        &layers[x - 1].affine,
                        &mut layers[x].affine
                    ) {
                        match cm.affine_type {
                            InsertLayer => {
                                if next_layer[y] == prev_layer[y] + cm.extend {
                                    //insertion gap instention from current affine layer
                                    x -= 1;
                                    save(&x, &y);
                                    break;
                                } else {
                                    // next_layer[j] == prev.m[j] + cm.open + cm.extend
                                    x -= 1;
                                    save(&x, &y);
                                    t = -1;
                                    break;
                                    //oppening new insertion gap from main layer
                                }
                            }
                            DeleteLayer => {
                                if next_layer[y] == next_layer[y - 1] + cm.extend {
                                    //deletion gap instention from current affine layer
                                    y -= 1;
                                    save(&x, &y);
                                    break;
                                } else {
                                    //next_layer[j] == next.m[j - 1] + cm.open + cm.extend
                                    y -= 1;
                                    save(&x, &y);
                                    t = -1;
                                    break;
                                }
                            }
                            _ => todo!(),
                        };
                    }
                } else {
                    x = 0;
                    y = 0;
                    save(&x, &y);
                }
            }
        }

        path.reverse();
        path
    }

    /// Computes the next layer (layer `i`) from the current one.
    /// `ca` is the `i-1`th character of sequence `a`.
    fn next_layer(
        &self,
        i: usize,
        ca: u8,
        b: &Sequence,
        prev: &NWLayers<N>,
        next: &mut NWLayers<N>,
        v: &mut impl Visualizer,
    ) {
        v.expand(Pos(i as I, 0));
        // Initialize the first state by linear insertion.
        next.m[0] = self.cm.ins_or(INF, |ins| i as Cost * ins);
        // Initialize the first state by affine insertion.
        for (cm, layer) in zip(&self.cm.affine, &mut next.affine) {
            match cm.affine_type {
                InsertLayer => {
                    layer[0] = cm.open + i as Cost * cm.extend;
                    next.m[0] = min(next.m[0], layer[0]);
                }
                DeleteLayer => {
                    layer[0] = INF;
                }
                _ => todo!(),
            };
        }
        for (j0, &cb) in b.iter().enumerate() {
            // Change from 0 to 1 based indexing.
            let j = j0 + 1;

            // Compute all layers at (i, j).
            v.expand(Pos(i as I, j as I));

            // Main layer: substitutions and linear indels.
            let mut f = INF;
            // NOTE: When sub/ins/del is not allowed, we have to skip them.
            if ca == cb {
                f = prev.m[j - 1];
            } else {
                if let Some(sub) = self.cm.sub {
                    f = min(f, prev.m[j - 1] + sub);
                }
            }
            if let Some(ins) = self.cm.ins {
                f = min(f, prev.m[j] + ins);
            }
            if let Some(del) = self.cm.del {
                f = min(f, next.m[j - 1] + del);
            }

            // Affine layers
            // TODO: Swap the order of this for loop and the loop over j?
            for (cm, prev_layer, next_layer) in
                izip!(&self.cm.affine, &prev.affine, &mut next.affine)
            {
                match cm.affine_type {
                    InsertLayer => {
                        next_layer[j] =
                            min(prev_layer[j] + cm.extend, prev.m[j] + cm.open + cm.extend)
                    }
                    DeleteLayer => {
                        next_layer[j] = min(
                            next_layer[j - 1] + cm.extend,
                            next.m[j - 1] + cm.open + cm.extend,
                        )
                    }
                    _ => todo!(),
                };
                f = min(f, next_layer[j]);
            }

            next.m[j] = f;
        }
    }
}

impl<const N: usize> Aligner for NW<AffineCost<N>> {
    /// The cost-only version uses linear memory.
    fn cost(&self, a: &Sequence, b: &Sequence) -> Cost {
        let ref mut prev = NWLayers::new(vec![INF; b.len() + 1]);
        let ref mut next = NWLayers::new(vec![INF; b.len() + 1]);

        next.m[0] = 0;
        for j in 1..=b.len() {
            // Initialize the main layer with linear deletions.
            next.m[j] = self.cm.del_or(INF, |del| j as Cost * del);

            // Initialize the affine deletion layers.
            for (cm, next_layer) in zip(&self.cm.affine, &mut next.affine) {
                match cm.affine_type {
                    DeleteLayer => {
                        next_layer[j] = cm.open + j as Cost * cm.extend;
                    }
                    InsertLayer => {}
                    _ => todo!(),
                };
                next.m[j] = min(next.m[j], next_layer[j]);
            }
        }

        for (i0, &ca) in a.iter().enumerate() {
            // Convert to 1 based index.
            let i = i0 + 1;
            std::mem::swap(prev, next);
            self.next_layer(i, ca, b, prev, next, &mut NoVisualizer);
        }

        return next.m[b.len()];
    }

    fn visualize(&self, a: &Sequence, b: &Sequence, v: &mut impl Visualizer) -> (Cost, PATH) {
        let ref mut layers = vec![NWLayers::<N>::new(vec![INF; b.len() + 1]); a.len() + 1];

        v.expand(Pos(0, 0));
        layers[0].m[0] = 0;
        for j in 1..=b.len() {
            v.expand(Pos(0, j as I));
            // Initialize the main layer with linear deletions.
            layers[0].m[j] = self.cm.del_or(INF, |del| j as Cost * del);

            // Initialize the affine deletion layers.
            let Layers { m, affine } = &mut layers[0];
            for (costs, next_layer) in izip!(&self.cm.affine, affine) {
                match costs.affine_type {
                    DeleteLayer => {
                        next_layer[j] = costs.open + j as Cost * costs.extend;
                    }
                    InsertLayer => {}
                    _ => todo!(),
                };
                m[j] = min(m[j], next_layer[j]);
            }
        }

        for (i0, &ca) in a.iter().enumerate() {
            // Change from 0-based to 1-based indexing.
            let i = i0 + 1;
            let [prev, next] = &mut layers[i-1..=i] else {unreachable!();};
            self.next_layer(i, ca, b, &*prev, next, v);
        }

        // FIXME: Backtrack the optimal path.

        let d = layers[a.len()].m[b.len()];
        return (d, track_path(layers, a, b));
    }
}
