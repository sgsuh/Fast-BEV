#include <cmath>
#include <cstdlib>
#include <iostream>

#include <libqhull_r/geom_r.h>
#include <libqhull_r/io_r.h>
#include <libqhull_r/poly_r.h>

#include "qhull_cus.h"

#define trace1(args) \
	{ }

typedef void (*printvridgeCusT)(qhT* qh, void* fp, vertexT* vertex, vertexT* vertexA, setT* centers, boolT unbounded);
extern int qh_eachvoronoi_all(qhT*, void* fp, printvridgeCusT printvridge, boolT is_upper, qh_RIDGE innerouter, boolT inorder);

QhullCus::QhullCus() {
}

QhullCus::QhullCus(std::string mode_option, std::vector<double>& points, std::string options, std::string required_options, bool incremental, bool furthest_site, std::vector<double> interior_point) {
	int exitcode;

	ndim_      = 2;
	numpoints_ = points.size() / 2;

	if(numpoints_ <= 0) {
		std::cerr << "No points given" << std::endl;
	}

	if(ndim_ < 2) {
		std::cerr << "Need at least 2-D data" << std::endl;
	}

	for(int i = 0; i < points.size(); ++i) {
		if(std::isnan(points[i])) {
			std::cerr << "Points cannot contain NaN" << std::endl;
		}
	}

	std::string options_c = "qhull " + mode_option + " " + options;

	point_arrays_ = points;

	is_delaunay_   = 0;
	is_halfspaces_ = 0;

	qh_ = (qhT*)std::malloc(sizeof(qhT));

	if(qh_ == NULL) {
		std::cout << "Memory allocation failed" << std::endl;
	}

	FILE* f = stderr;

	qh_zero(qh_, f);

	coordT* coord = NULL;

	exitcode = qhNewQhullScipy(qh_, ndim_, numpoints_, (realT*)points.data(), 0, (char*)options_c.c_str(), NULL, NULL, coord);

	if(exitcode != 0) {
		close();
	}
}

QhullCus::~QhullCus() {
	int curlong;
	int totlong;

	if(qh_ != NULL) {
		qh_freeqhull(qh_, qh_ALL);
		qh_memfreeshort(qh_, &curlong, &totlong);
		std::free(qh_);

		qh_ = NULL;

		vertices_.clear();

		if(curlong != 0 || totlong != 0) {
			std::cerr << "qhull: did not free " << totlong << " bytes (" << curlong << " pieces)" << std::endl;
		}
	}
}

int QhullCus::qhNewQhullScipy(qhT* _qh, int dim, int numpoints, coordT* points, boolT ismalloc, char* qhull_cmd, void* outfile, void* errfile, coordT* feaspoint) {
	int exitcode;
	int hulldim;

	coordT* new_points;
	boolT   new_ismalloc;

	if(!errfile) {
		errfile = stderr;
	}

	if(!_qh->qhmem.ferr) {
		qh_meminit(_qh, (FILE*)errfile);
	} else {
		qh_memcheck(_qh);
	}

	qh_initqhull_start(_qh, NULL, (FILE*)outfile, (FILE*)errfile);

	trace1((_qh, _qh->ferr, 1044, "qh_new_qhull: build new Qhull for %d %d-d points with %s\n", numpoints, dim, qhull_cmd));

	exitcode = setjmp(_qh->errexit);

	if(!exitcode) {
		_qh->NOerrexit = false;

		qh_initflags(_qh, (char*)"");

		if(_qh->DELAUNAY) {
			_qh->PROJECTdelaunay = true;
		}

		if(_qh->HALFspace) {
			hulldim = dim - 1;

			if(feaspoint) {
				coordT* coords = _qh->feasible_point;
				coordT* value  = feaspoint;

				for(int i = 0; i < hulldim; ++i) {
					*(coords++) = *(value++);
				}

			} else {
				qh_setfeasible(_qh, hulldim);
			}

			new_points   = qh_sethalfspace_all(_qh, dim, numpoints, points, _qh->feasible_point);
			new_ismalloc = true;

			if(ismalloc) {
				qh_free(points);
			}
		} else {
			hulldim      = dim;
			new_points   = points;
			new_ismalloc = ismalloc;
		}

		qh_init_B(_qh, new_points, numpoints, hulldim, new_ismalloc);
		qh_qhull(_qh);
		qh_check_output(_qh);

		if(outfile) {
			qh_produce_output(_qh);
		} else {
			qh_prepare_output(_qh);
		}

		vertices_ = getExtreams2d();
		va_       = volumeArea();

		if(_qh->VERIFYoutput && !_qh->FORCEoutput && !_qh->STOPadd && !_qh->STOPcone && !_qh->STOPpoint) {
			qh_check_points(_qh);
		}
	}

	_qh->NOerrexit = true;

	return exitcode;
}

void QhullCus::close() {
	int curlong;
	int totlong;

	if(qh_ != NULL) {
		qh_freeqhull(qh_, qh_ALL);
		qh_memfreeshort(qh_, &curlong, &totlong);
		std::free(qh_);

		qh_ = NULL;

		if(curlong != 0 || totlong != 0) {
			std::cerr << "qhull: did not free " << totlong << " bytes (" << curlong << " pieces)" << std::endl;
		}
	}
}

std::vector<double> QhullCus::getPoints() {
	return point_arrays_;
}

void QhullCus::checkActive() {
	if(qh_ == NULL) {
		std::cerr << "Qhull instance is closed" << std::endl;
	}
}

void QhullCus::addPoints(std::vector<double>& points, std::vector<double> interior_point) {
	checkActive();

	int     exitcode;
	realT*  p;
	facetT* facet;
	double  bestdist;
	boolT   isoutside;

	try {
		exitcode = setjmp(qh_[0].errexit);

		qh_[0].NOerrexit = 0;

		p = (realT*)points.data();

		for(int j = 0; j < points.size(); j += 2) {
			facet = qh_findbestfacet(qh_, p, 0, &bestdist, &isoutside);

			if(isoutside) {
				if(!qh_addpoint(qh_, p, facet, 0)) {
					break;
				}
			} else {
				qh_setappend(qh_, &qh_[0].other_points, p);
			}

			p += 2;
		}

		qh_check_maxout(qh_);

		qh_[0].hasTriangulation = 0;

		for(int j = 0; j < points.size(); ++j) {
			point_arrays_.emplace_back(points[j]);
		}

		numpoints_ += int(points.size() / 2);

		qh_findgood_all(qh_, qh_[0].facet_list);

	} catch(...) {
		qh_[0].NOerrexit = 1;
	}
}

std::array<double, 2> QhullCus::getParaboloidShiftScale() {
	std::array<double, 2> paraboloid;  // 0: Scale, 1: Shift

	checkActive();

	if(qh_[0].SCALElast) {
		paraboloid[0] = qh_[0].last_newhigh / (qh_[0].last_high - qh_[0].last_low);
		paraboloid[1] = qh_[0].last_low * paraboloid[0];
	} else {
		paraboloid[0] = 1.0f;
		paraboloid[1] = 0.0f;
	}

	return paraboloid;
}

std::array<double, 2> QhullCus::volumeArea() {
	std::array<double, 2> va;  // 0: Volume, 1: Area

	checkActive();

	qh_->hasAreaVolume = 0;

	qh_getarea(qh_, qh_[0].facet_list);

	va[0] = qh_[0].totvol;
	va[1] = qh_[0].totarea;

	return va;
}

void QhullCus::triangulate() {
	checkActive();

	qh_triangulate(qh_);
}

void QhullCus::getSimplexFacetArray(std::vector<int>& facets, std::vector<int>& neighbors, std::vector<double>& equations, std::vector<int>& coplanar, std::vector<int>& good) {
	checkActive();

	int facet_ndim = ndim_;

	std::vector<int> id_map(qh_[0].facet_id);

	// Compute facet indices
	for(int i = 0; i < qh_[0].facet_id; ++i) {
		id_map[i] = -1;
	}

	facetT* facet = qh_[0].facet_list;

	int j = 0;

	while(facet && facet->next) {
		if(!is_delaunay_ || facet->upperdelaunay == qh_[0].UPPERdelaunay) {
			if(!facet->simplicial && (qh_setsize(qh_, facet->vertices) != facet_ndim || qh_setsize(qh_, facet->neighbors) != facet_ndim)) {
				std::cerr << "non-simplical facet encounterd: " << qh_setsize(qh_, facet->vertices) << " vertices" << std::endl;
			}

			id_map[facet->id] = j;
			j++;
		}

		facet = facet->next;
	}

	// Allocate output
	facets.resize(j * facet_ndim);
	good.resize(j);
	neighbors.resize(j * facet_ndim);
	equations.resize(j * (facet_ndim + 1));

	int ncoplanar = 0;
	coplanar.resize(10 * 3);

	// Retrieve facet information
	facet = qh_[0].facet_list;
	j     = 0;

	vertexT* vertex;
	int      ipoint;
	facetT*  neighbor;

	while(facet && facet->next) {
		if(is_delaunay_ && facet->upperdelaunay != qh_[0].UPPERdelaunay) {
			facet = facet->next;

			continue;
		}

		// Use a lower bound so that the tight loop in high dimensions is not affected by the conditional below
		unsigned int lower_bound = 0;

		if(is_delaunay_ && facet->toporient == qh_ORIENTclock && facet_ndim == 3) {
			// Swap the first and second indices to maintain a counter-clockwise orientation
			for(int i = 0; i < 2; ++i) {
				// Save the vertex info
				unsigned int swapped_index             = 1 ^ i;
				vertex                                 = (vertexT*)facet->vertices->e[i].p;
				ipoint                                 = qh_pointid(qh_, vertex->point);
				facets[j * facet_ndim + swapped_index] = ipoint;

				// Save the neighbor info
				neighbor                                  = (facetT*)facet->neighbors->e[i].p;
				neighbors[j * facet_ndim + swapped_index] = id_map[neighbor->id];
			}

			lower_bound = 2;
		}

		for(int i = lower_bound; i < facet_ndim; ++i) {
			// Save the vertex info
			vertex                     = (vertexT*)facet->vertices->e[i].p;
			ipoint                     = qh_pointid(qh_, vertex->point);
			facets[j * facet_ndim + i] = ipoint;

			// Save the neighbor info
			neighbor                      = (facetT*)facet->neighbors->e[i].p;
			neighbors[j * facet_ndim + i] = id_map[neighbor->id];
		}

		// Save simplex equation info
		for(int i = 0; i < facet_ndim; ++i) {
			equations[j * (facet_ndim + 1) + i] = facet->normal[i];
		}

		equations[j * (facet_ndim + 1) + facet_ndim] = facet->offset;

		// Save coplanar info
		if(facet->coplanarset) {
			for(int i = 0; i < qh_setsize(qh_, facet->coplanarset); ++i) {
				pointT* point = (pointT*)facet->coplanarset->e[i].p;
				double  dist;
				vertex = qh_nearvertex(qh_, facet, point, &dist);

				if(ncoplanar >= 10) {
					coplanar.emplace_back(qh_pointid(qh_, point));
					coplanar.emplace_back(id_map[facet->id]);
					coplanar.emplace_back(qh_pointid(qh_, vertex->point));
				} else {
					coplanar[ncoplanar * 3 + 0] = qh_pointid(qh_, point);
					coplanar[ncoplanar * 3 + 1] = id_map[facet->id];
					coplanar[ncoplanar * 3 + 2] = qh_pointid(qh_, vertex->point);
				}

				ncoplanar += 1;
			}
		}

		// Save good info
		good[j] = facet->good;

		j += 1;
		facet = facet->next;
	}
}

std::vector<double> QhullCus::getHullPoints() {
	checkActive();

	int point_ndim = ndim_;
	int numpoints  = qh_->num_points;

	std::vector<double> points(numpoints * point_ndim);

	pointT* point;

	for(int i = 0; i < numpoints; ++i) {
		for(int j = 0; j < point_ndim; ++j) {
			points[i * point_ndim + j] = point[j];
		}

		point += qh_->hull_dim;
	}

	return points;
}

void QhullCus::getHullFacets(std::vector<std::vector<int>>& facets, std::vector<double>& equations) {
	checkActive();

	int facet_ndim = ndim_;
	int numfacets  = qh_->num_facets - qh_->num_visible;

	facetT* facet = qh_->facet_list;

	equations.resize(numfacets * (facet_ndim + 1));

	int i = 0;

	while(facet && facet->next) {
		std::vector<int> facetsi;

		int j = 0;

		for(j = 0; j < facet_ndim; ++j) {
			equations[i * (facet_ndim + 1) + j] = facet->normal[j];
		}

		equations[i * (facet_ndim + 1) + facet_ndim] = facet->offset;

		j = 0;

		vertexT* vertex = (vertexT*)facet->vertices->e[0].p;

		while(vertex) {
			// Save the vertex info
			int ipoint = qh_pointid(qh_, vertex->point);

			facetsi.emplace_back(ipoint);

			j += 1;

			vertex = (vertexT*)facet->vertices->e[j].p;
		}

		i += 1;

		facets.emplace_back(facetsi);

		facet = facet->next;
	}
}

void visitVoronoi(qhT* _qh, void* ptr, vertexT* vertex, vertexT* vertex_a, setT* centers, boolT unbounded) {
	QhullCus* qh_cus = (QhullCus*)ptr;

	return;
}

void QhullCus::getVoronoiDiagram(std::vector<double>& voronoiVertices, std::vector<std::vector<int>>& regions, std::vector<int>& point_region) {
	checkActive();

	nridges_ = 0;

	ridge_points_.clear();
	ridge_points_.resize(10 * 2);

	ridge_vertices_.clear();

	// TODO
}

std::vector<int> QhullCus::getExtreams2d() {
	checkActive();

	int              nextremes = 0;
	std::vector<int> extremes_arr(100);
	std::vector<int> extremes = extremes_arr;

	qh_->visit_id += 1;
	qh_->vertex_visit += 1;

	facetT* facet      = qh_->facet_list;
	facetT* startfacet = facet;
	facetT* nextfacet;

	vertexT* vertex_a;
	vertexT* vertex_b;

	while(facet) {
		if(facet->toporient) {
			vertex_a  = (vertexT*)facet->vertices->e[0].p;
			vertex_b  = (vertexT*)facet->vertices->e[1].p;
			nextfacet = (facetT*)facet->neighbors->e[0].p;
		} else {
			vertex_b  = (vertexT*)facet->vertices->e[0].p;
			vertex_a  = (vertexT*)facet->vertices->e[1].p;
			nextfacet = (facetT*)facet->neighbors->e[1].p;
		}

		if(nextremes + 2 >= extremes.size()) {
			extremes.clear();

			// Array is safe to resize
			extremes_arr.resize(2 * extremes_arr.size() + 1);
			extremes = extremes_arr;
		}

		if(vertex_a->visitid != qh_->vertex_visit) {
			vertex_a->visitid   = qh_->vertex_visit;
			extremes[nextremes] = qh_pointid(qh_, vertex_b->point);
			nextremes += 1;
		}

		facet->visitid = qh_->visit_id;
		facet          = nextfacet;

		if(facet == startfacet) {
			break;
		}
	}

	// This array is always safe to resize

	extremes.resize(nextremes);

	return extremes;
}

QhullUser::QhullUser(QhullCus qhull, bool incremental) {
	update(qhull);

	qhull_ = qhull;
}

void QhullUser::update(QhullCus& qhull) {
	points_  = qhull.getPoints();
	ndim_    = qhull.ndim_;
	npoints_ = points_.size() / ndim_;

	min_bound_.resize(ndim_);
	max_bound_.resize(ndim_);

	for(int i = 0; i < ndim_; ++i) {
		min_bound_[i] = DBL_MAX;
		max_bound_[i] = -DBL_MAX;
	}

	for(int i = 0; i < npoints_; ++i) {
		for(int j = 0; j < ndim_; ++j) {
			if(points_[i * ndim_ + j] < min_bound_[j]) {
				min_bound_[j] = points_[i * ndim_ + j];
			}

			if(points_[i * ndim_ + j] > max_bound_[j]) {
				max_bound_[j] = points_[i * ndim_ + j];
			}
		}
	}
}

void QhullUser::addPoints(std::vector<double>& points, bool restart, std::vector<double> interior_point) {
	qhull_.addPoints(points, interior_point);
	update(qhull_);
}

ConvexHull::ConvexHull(std::vector<double>& points, bool incremental, std::string qhull_options)
    : QhullUser(QhullCus("i", points, qhull_options, "Qt", incremental), incremental) {
}

void ConvexHull::update(QhullCus& qhull) {
	qhull.triangulate();

	qhull.getSimplexFacetArray(simplices_, neighbors_, equations_, coplanar_, good_);

	std::array<double, 2> va = qhull.volumeArea();

	volume_ = va[0];
	area_   = va[1];

	if(qhull.ndim_ == 2) {
		vertices_ = qhull.getExtreams2d();
	} else {
		vertices_.clear();
	}

	nsimplex_ = simplices_.size() / qhull.ndim_;

	QhullUser::update(qhull);
}

void ConvexHull::addPoints(std::vector<double>& points, bool restart) {
	QhullUser::addPoints(points, restart);
}

std::vector<double> ConvexHull::points() {
	return points_;
}

std::vector<int> ConvexHull::vertices() {
	return qhull_.vertices_;
}

float ConvexHull::volume() {
	return qhull_.va_[0];
}
