/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Claire Brecheteau
 *
 *    Copyright (C) 2021 Université Rennes 2
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 *
 *	A ECRIRE UNE FOIS QUE TOUT EST FINI - INSPIRE DE GUDHI !!!
 */

#ifndef DOC_KPLM_H_
#define DOC_KPLM_H_

namespace Gudhi {

namespace kplm {

/**  \defgroup kplm_algorithm kPLM algorithm
 * 
 * \author    Claire Brecheteau
 * 
 * @{
 * 
 * This module implements a clustering method of k-means-type, based on Mahalanobis distances.
 * In particular, to this method is associated a distance function than can be considered as a robust approximation 
 * of the distance to the underlying support of the sample points.
 * 
 * \section edge_collapse_definition Edge collapse definition
 * 
 * An edge \f$e\f$ in a simplicial complex \f$K\f$ is called a <b>dominated edge</b> if 
 * 
 * -- Blabla \f$e \in K\f$
 * 
 * -- Blibli \f$e \in P\f$
 * 
 * 
 * \subsection kPLMexample Basic example for the kPLM
 * 
 * This example calls `Gudhi::collapse::flag_complex_collapse_edges()` from a proximity graph represented as a list of
 * `Filtered_edge`.
 * Then it collapses edges and displays a new list of `Filtered_edge` (with less edges)
 * that will preserve the persistence homology computation.
 * 
 * \include Collapse/edge_collapse_basic_example.cpp
 * 
 * When launching the example:
 * 
 * \code $> ./Edge_collapse_example_basic
 * \endcode
 *
 * the program output is:
 * 
 * \include Collapse/edge_collapse_example_basic.txt
 * \image html "funcGICvisu.jpg" "Visualization with neato"
 * \include Nerve_GIC/CoordGIC.cpp
 *
 * When launching:
 *
 * \code $> ./CoordGIC ../../data/points/KleinBottle5D.off 0 -v
 * \endcode
 *
 * the program outputs SC.dot. Using e.g.
 *
 * \code $> neato SC.dot -Tpdf -o SC.pdf
 * \endcode
 *
 * one can obtain the following visualization:
 *
 * \image html "coordGICvisu2.jpg" "Visualization with Neato"
 */
/** @} */  // end defgroup kplm_algorithm

}  // namespace kplm

}  // namespace Gudhi

#endif  // DOC_KPLM_H_
