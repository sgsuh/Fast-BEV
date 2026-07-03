#pragma once

#include <array>
#include <vector>

#include <libqhull_r/libqhull_r.h>

class QhullCus {
   public:
	QhullCus();
	QhullCus(std::string mode_option, std::vector<double>& points, std::string options = "", std::string required_options = "", bool incremental = false, bool furthest_site = false, std::vector<double> interior_point = {});
	virtual ~QhullCus();

	void triangulate();
	void getSimplexFacetArray(std::vector<int>& facets, std::vector<int>& neighbors, std::vector<double>& equations, std::vector<int>& coplanar, std::vector<int>& good);
	void addPoints(std::vector<double>& points, std::vector<double> interior_point = {});

	std::vector<double> getPoints();
	std::vector<int>    getExtreams2d();

	std::array<double, 2> volumeArea();

	void close();

	int ndim_;

	qhT* qh_;

	std::vector<int>      vertices_;
	std::array<double, 2> va_ = {0.0f, 0.0f};

   protected:
	bool is_delaunay_;
	bool is_halfspaces_;

	int numpoints_;
	int nridges_;

	std::vector<int> ridge_points_;

	std::vector<double> point_arrays_;

	std::vector<std::vector<int>> ridge_vertices_;

	void checkActive();

	void getHullFacets(std::vector<std::vector<int>>& facets, std::vector<double>& equations);
	void getVoronoiDiagram(std::vector<double>& voronoiVertices, std::vector<std::vector<int>>& regions, std::vector<int>& point_region);

	int qhNewQhullScipy(qhT*, int dim, int numpoints, coordT* points, boolT ismalloc, char* qhull_cmd, void* outfile, void* errfile, coordT* feaspoint);

	std::vector<double> getHullPoints();

	std::array<double, 2> getParaboloidShiftScale();
};

class QhullUser {
   public:
	QhullUser(QhullCus qhull, bool incremental = false);

	void update(QhullCus& qhull);
	void addPoints(std::vector<double>& points, bool restart = false, std::vector<double> interior_point = {});

	std::vector<double> points_;

   protected:
	QhullCus qhull_;

	int                 ndim_;
	int                 npoints_;
	std::vector<double> min_bound_;
	std::vector<double> max_bound_;
};

class ConvexHull: public QhullUser {
   public:
	ConvexHull(std::vector<double>& points, bool incremental = false, std::string qhull_options = "");

	std::vector<int> vertices();
	float            volume();

	double volume_;
	double area_;

   protected:
	void update(QhullCus& qhull);
	void addPoints(std::vector<double>& points, bool restart = false);

	std::vector<double> points();

	int nsimplex_;

	std::vector<int>    simplices_;
	std::vector<int>    neighbors_;
	std::vector<double> equations_;
	std::vector<int>    coplanar_;
	std::vector<int>    good_;
	std::vector<int>    vertices_;
};
