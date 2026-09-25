#pragma once
#include <memory>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

namespace ccann {

  class Parameters {
   public:
    Parameters() { Set<int>("num_threads", 0); }

    template<typename ParamType>
    inline void Set(const std::string &name, const ParamType &value) {
      params[name] = std::shared_ptr<void>(new ParamType(value), [](void *ptr) {
        delete static_cast<ParamType *>(ptr);
      });
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        // return ConvertStrToValue<ParamType>(item->second);
        if (item->second == nullptr) {
          throw std::invalid_argument(std::string("Parameter ") + name + " has value null.");
        } else {
          return *(static_cast<ParamType *>(item->second.get()));
        }
      }
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name, const ParamType &default_value) {
      try {
        return Get<ParamType>(name);
      } catch (std::invalid_argument e) {
        return default_value;
      }
    }

    ~Parameters() = default;

   private:
    std::unordered_map<std::string, std::shared_ptr<void>> params;

    Parameters(const Parameters &);
    Parameters &operator=(const Parameters &);

    template<typename ParamType>
    inline ParamType ConvertStrToValue(const std::string &str) const {
      std::stringstream sstream(str);
      ParamType value;
      if (!(sstream >> value) || !sstream.eof()) {
        std::stringstream err;
        err << "Failed to convert value '" << str << "' to type: " << typeid(value).name();
        throw std::runtime_error(err.str());
      }
      return value;
    }
  };
}  // namespace ccann
