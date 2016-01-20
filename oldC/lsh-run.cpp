
/**
\verbatim
Allowed options:
  -h [ --help ]            produce help message.
  -M [ -- ] arg (=20)      number of repeats for RepeatHash
  -L [ -- ] arg (=1)       number of hash tables
  -Q [ -- ] arg (=100)     number of queries to use
  -K [ -- ] arg (=50)      number of nearest neighbors to retrieve
  -D [ --data ] arg        dataset path
  -H [ -- ] arg (=1017881) hash table size, use the default.
\endverbatim
**/

#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <lshkit.h>

using namespace std;
using namespace lshkit;
namespace po = boost::program_options; 

int main (int argc, char *argv[])
{
    string data_file;
    string query_file;
    string output_file;

    float R;
    unsigned M, L, H;
    unsigned K, W;

    boost::timer timer;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        (",M", po::value<unsigned>(&M)->default_value(20), "number of repeats for RepeatHash")
        (",L", po::value<unsigned>(&L)->default_value(1), "number of hash tables")
        (",W", po::value<unsigned>(&W)->default_value(100), "window size to use")
        (",K", po::value<unsigned>(&K)->default_value(50), "number of nearest neighbors to retrieve")
        (",R", po::value<float>(&R)->default_value(numeric_limits<float>::max()), "R-NN distance range")
        ("data,D", po::value<string>(&data_file), "dataset path")
        ("query,Q", po::value<string>(&query_file), "queryset path")
        ("output,O", po::value<string>(&output_file), "output file and path")
        (",H", po::value<unsigned>(&H)->default_value(1017881), "hash table size, use the default.")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 

    if (vm.count("help") || (vm.count("data") < 1) || (vm.count("query") < 1) || (vm.count("output") < 1))
    {
        cout << desc;
        return 0;
    }

    cout << "LOADING DATA..." << endl;
    timer.restart();
    
    FloatMatrix data(data_file); //Uses memory map under the covers
    cout << boost::format("LOAD TIME: %1%s.") % timer.elapsed() << endl;

    cout << "LOADING QUERY..." << endl;
    timer.restart();
    
    FloatMatrix query(query_file); //Uses memory map under the covers
    cout << boost::format("LOAD TIME: %1%s.") % timer.elapsed() << endl;
    
    typedef Tail<RepeatHash<GaussianLsh> > MyLsh;
    typedef LshIndex<MyLsh, unsigned> Index;

    //Use L2 metric
    metric::l2<float> l2(data.getDim());
    FloatMatrix::Accessor accessor(data);
    Index index;

    Index::Parameter param;

    // Setup the parameters.  Note that L is not provided here.
    param.range = H;
    param.repeat = M;
    param.W = W;
    param.dim = data.getDim();
    DefaultRng rng;
    index.init(param, rng, L);

    // Initialize the index structure.  Note L is passed here.
    cout << "CONSTRUCTING INDEX..." << endl;

    timer.restart();
    {
        boost::progress_display progress(data.getSize());
        for (unsigned i = 0; i < data.getSize(); ++i)
        {
            // Insert an item to the hash table.
            // Note that only the key is passed in here.
            // MPLSH will get the feature from the accessor.
            index.insert(i, data[i]);
            ++progress;
        }
    }
    cout << boost::format("CONSTRUCTION TIME: %1%s.") % timer.elapsed() << endl;

    cout << "RUNNING QUERIES..." << endl;
    ofstream myfile;
    myfile.open(output_file.c_str());
    
    timer.restart();
    {
        TopkScanner<FloatMatrix::Accessor, metric::l2<float> > queryScanner(accessor, l2, K, R);
        boost::progress_display progress(query.getSize());
        for (unsigned i = 0; i < query.getSize(); ++i)
        {
            queryScanner.reset(query[i]);
            index.query(query[i], queryScanner);
            //Write out to file
            myfile << i << ':';
            for (unsigned ii = 0; ii < queryScanner.topk().size(); ++ii) {
                myfile << queryScanner.topk()[ii].key << ':' << queryScanner.topk()[ii].dist << ':';
            }
            myfile << endl;
            ++progress;
        }
    }
    myfile.close();
    cout << boost::format("QUERY TIME: %1%s.") % timer.elapsed() << endl;

    return 0;
}

